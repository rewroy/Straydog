//! Host decoding of every GGUF block type equals ggml's definition (ggml-quants.c, MIT): each
//! `dequantize_row_*` below is a line-for-line port, run on real blocks of the DeepSeek-V4.1 and
//! Qwen3.8 bundles (tests/golden/format_blocks.json; no MXFP4 file is on hand, so MXFP4 is tested
//! on generated blocks only) and on generated blocks of every type, compared bit for bit.
//!
//! ggml's C may contract `dl * q - ml` into a fused multiply-add; its products are exact, so the
//! fused and unfused forms round alike and the port's plain arithmetic is the definition.

use std::path::PathBuf;

use serde_json::Value as Json;
use straydog_engine::formats::{self, f16_to_f32, iq_grid};

const KSIGNS_IQ2XS: [u8; 128] = [
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
];
const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
const KVALUES_IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
const IQ1S_DELTA: f32 = 0.125;
const QK_K: usize = 256;

fn fp16(x: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([x[at], x[at + 1]]))
}
fn u16_at(x: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([x[at], x[at + 1]])
}
fn u32_at(x: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(x[at..at + 4].try_into().unwrap())
}
fn sign(signs: u8, j: usize) -> f32 {
    if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 }
}
/// ggml_e8m0_to_fp32_half.
fn e8m0_half(x: u8) -> f32 {
    f32::from_bits(if x < 2 { 0x0020_0000 << x } else { u32::from(x - 1) << 23 })
}
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// One block of `kind` as ggml's dequantize_row_* decodes it.
fn ggml_dequantize(kind: u32, x: &[u8]) -> Vec<f32> {
    let mut y = Vec::new();
    match kind {
        0 => y.extend(x.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap()))),
        1 => y.extend(x.chunks_exact(2).map(|b| fp16(b, 0))),
        30 => y.extend(x.chunks_exact(2).map(|b| f32::from_bits(u32::from(u16_at(b, 0)) << 16))),
        2 => {   // dequantize_row_q4_0
            let d = fp16(x, 0);
            y.resize(32, 0.0);
            for j in 0..16 {
                let x0 = i32::from(x[2 + j] & 0x0F) - 8;
                let x1 = i32::from(x[2 + j] >> 4) - 8;
                y[j] = x0 as f32 * d;
                y[j + 16] = x1 as f32 * d;
            }
        }
        6 => {   // dequantize_row_q5_0
            let d = fp16(x, 0);
            let qh = u32_at(x, 2);
            y.resize(32, 0.0);
            for j in 0..16 {
                let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
                let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;
                let x0 = i32::from((x[6 + j] & 0x0F) | xh_0) - 16;
                let x1 = i32::from((x[6 + j] >> 4) | xh_1) - 16;
                y[j] = x0 as f32 * d;
                y[j + 16] = x1 as f32 * d;
            }
        }
        8 => {   // dequantize_row_q8_0
            let d = fp16(x, 0);
            y.extend((0..32).map(|j| f32::from(x[2 + j] as i8) * d));
        }
        20 => {   // dequantize_row_iq4_nl
            let d = fp16(x, 0);
            y.resize(32, 0.0);
            for j in 0..16 {
                y[j] = d * f32::from(KVALUES_IQ4NL[usize::from(x[2 + j] & 0xf)]);
                y[j + 16] = d * f32::from(KVALUES_IQ4NL[usize::from(x[2 + j] >> 4)]);
            }
        }
        39 => {   // dequantize_row_mxfp4: e, qs[16]
            let d = e8m0_half(x[0]);
            y.resize(32, 0.0);
            for j in 0..16 {
                let x0 = KVALUES_MXFP4[usize::from(x[1 + j] & 0x0F)];
                let x1 = KVALUES_MXFP4[usize::from(x[1 + j] >> 4)];
                y[j] = f32::from(x0) * d;
                y[j + 16] = f32::from(x1) * d;
            }
        }
        10 => {   // dequantize_row_q2_K: scales[16] qs[64] d dmin
            let d = fp16(x, 80);
            let min = fp16(x, 82);
            let mut q = 16;
            let mut is = 0;
            for _n in (0..QK_K).step_by(128) {
                let mut shift = 0;
                for _j in 0..4 {
                    let mut sc = x[is];
                    is += 1;
                    let (mut dl, mut ml) = (d * f32::from(sc & 0xF), min * f32::from(sc >> 4));
                    for l in 0..16 {
                        y.push(dl * f32::from(((x[q + l] >> shift) & 3) as i8) - ml);
                    }
                    sc = x[is];
                    is += 1;
                    (dl, ml) = (d * f32::from(sc & 0xF), min * f32::from(sc >> 4));
                    for l in 0..16 {
                        y.push(dl * f32::from(((x[q + l + 16] >> shift) & 3) as i8) - ml);
                    }
                    shift += 2;
                }
                q += 32;
            }
        }
        11 => {   // dequantize_row_q3_K: hmask[32] qs[64] scales[12] d
            let d_all = fp16(x, 108);
            let kmask1 = 0x0303_0303u32;
            let kmask2 = 0x0f0f_0f0fu32;
            let mut aux = [u32_at(x, 96), u32_at(x, 100), u32_at(x, 104), 0];
            let tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | ((tmp & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            let scales: Vec<i8> = aux.iter().flat_map(|w| w.to_le_bytes()).map(|b| b as i8).collect();
            let mut q = 32;
            let mut is = 0;
            let mut m = 1u8;
            for _n in (0..QK_K).step_by(128) {
                let mut shift = 0;
                for _j in 0..4 {
                    let mut dl = d_all * f32::from(scales[is] - 32);
                    is += 1;
                    for l in 0..16 {
                        let high = if x[l] & m != 0 { 0 } else { 4 };
                        y.push(dl * f32::from(((x[q + l] >> shift) & 3) as i8 - high));
                    }
                    dl = d_all * f32::from(scales[is] - 32);
                    is += 1;
                    for l in 0..16 {
                        let high = if x[l + 16] & m != 0 { 0 } else { 4 };
                        y.push(dl * f32::from(((x[q + l + 16] >> shift) & 3) as i8 - high));
                    }
                    shift += 2;
                    m <<= 1;
                }
                q += 32;
            }
        }
        12 | 13 => {   // dequantize_row_q4_K / q5_K: d dmin scales[12] (qh[32]) qs[128]
            let d = fp16(x, 0);
            let min = fp16(x, 2);
            let scales = &x[4..16];
            let mut ql = if kind == 13 { 48 } else { 16 };
            let (mut is, mut u1, mut u2) = (0, 1u8, 2u8);
            for _j in (0..QK_K).step_by(64) {
                let (sc, m) = get_scale_min_k4(is, scales);
                let (d1, m1) = (d * f32::from(sc), min * f32::from(m));
                let (sc, m) = get_scale_min_k4(is + 1, scales);
                let (d2, m2) = (d * f32::from(sc), min * f32::from(m));
                for l in 0..32 {
                    let high = if kind == 13 && x[16 + l] & u1 != 0 { 16 } else { 0 };
                    y.push(d1 * f32::from((x[ql + l] & 0xF) + high) - m1);
                }
                for l in 0..32 {
                    let high = if kind == 13 && x[16 + l] & u2 != 0 { 16 } else { 0 };
                    y.push(d2 * f32::from((x[ql + l] >> 4) + high) - m2);
                }
                ql += 32;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        14 => {   // dequantize_row_q6_K: ql[128] qh[64] scales[16] d
            let d = fp16(x, 208);
            y.resize(QK_K, 0.0);
            let (mut out, mut ql, mut qh, mut sc) = (0, 0, 128, 192);
            for _n in (0..QK_K).step_by(128) {
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((x[ql + l] & 0xF) | ((x[qh + l] & 3) << 4)) as i8 - 32;
                    let q2 = ((x[ql + l + 32] & 0xF) | (((x[qh + l] >> 2) & 3) << 4)) as i8 - 32;
                    let q3 = ((x[ql + l] >> 4) | (((x[qh + l] >> 4) & 3) << 4)) as i8 - 32;
                    let q4 = ((x[ql + l + 32] >> 4) | (((x[qh + l] >> 6) & 3) << 4)) as i8 - 32;
                    let s = |k: usize| f32::from(x[sc + is + k] as i8);
                    y[out + l] = d * s(0) * f32::from(q1);
                    y[out + l + 32] = d * s(2) * f32::from(q2);
                    y[out + l + 64] = d * s(4) * f32::from(q3);
                    y[out + l + 96] = d * s(6) * f32::from(q4);
                }
                out += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        16 => {   // dequantize_row_iq2_xxs: d qs[32] (u16)
            let d = fp16(x, 0);
            let grid_table = iq_grid(0);
            for ib32 in 0..QK_K / 32 {
                let aux32 = [u32_at(x, 2 + 8 * ib32), u32_at(x, 6 + 8 * ib32)];
                let aux8 = aux32[0].to_le_bytes();
                let db = d * (0.5 + (aux32[1] >> 28) as f32) * 0.25;
                for l in 0..4 {
                    let grid = &grid_table[8 * usize::from(aux8[l])..];
                    let signs = KSIGNS_IQ2XS[((aux32[1] >> (7 * l)) & 127) as usize];
                    y.extend((0..8).map(|j| db * f32::from(grid[j]) * sign(signs, j)));
                }
            }
        }
        17 => {   // dequantize_row_iq2_xs: d qs[32] (u16) scales[8]
            let d = fp16(x, 0);
            let grid_table = iq_grid(1);
            for ib32 in 0..QK_K / 32 {
                let scale = x[66 + ib32];
                let db = [d * (0.5 + f32::from(scale & 0xf)) * 0.25, d * (0.5 + f32::from(scale >> 4)) * 0.25];
                for l in 0..4 {
                    let q = u16_at(x, 2 + 2 * (4 * ib32 + l));
                    let grid = &grid_table[8 * usize::from(q & 511)..];
                    let signs = KSIGNS_IQ2XS[usize::from(q >> 9)];
                    y.extend((0..8).map(|j| db[l / 2] * f32::from(grid[j]) * sign(signs, j)));
                }
            }
        }
        22 => {   // dequantize_row_iq2_s: d qs[64] (grid bytes, then signs) qh[8] scales[8]
            let d = fp16(x, 0);
            let grid_table = iq_grid(2);
            let (mut qs, qh, mut signs) = (2, 66, 34);
            for ib32 in 0..QK_K / 32 {
                let scale = x[74 + ib32];
                let db = [d * (0.5 + f32::from(scale & 0xf)) * 0.25, d * (0.5 + f32::from(scale >> 4)) * 0.25];
                for l in 0..4 {
                    let dl = db[l / 2];
                    let index = usize::from(x[qs + l]) | ((usize::from(x[qh + ib32]) << (8 - 2 * l)) & 0x300);
                    let grid = &grid_table[8 * index..];
                    y.extend((0..8).map(|j| dl * f32::from(grid[j]) * sign(x[signs + l], j)));
                }
                qs += 4;
                signs += 4;
            }
        }
        18 => {   // dequantize_row_iq3_xxs: d qs[64] scales_and_signs[32]
            let d = fp16(x, 0);
            let grid_table = iq_grid(3);
            let mut qs = 2;
            for ib32 in 0..QK_K / 32 {
                let aux32 = u32_at(x, 66 + 4 * ib32);
                let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
                for l in 0..4 {
                    let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                    let grid1 = &grid_table[4 * usize::from(x[qs + 2 * l])..];
                    let grid2 = &grid_table[4 * usize::from(x[qs + 2 * l + 1])..];
                    let mut values = [0f32; 8];
                    for j in 0..4 {
                        values[j] = db * f32::from(grid1[j]) * sign(signs, j);
                        values[j + 4] = db * f32::from(grid2[j]) * sign(signs, j + 4);
                    }
                    y.extend(values);
                }
                qs += 8;
            }
        }
        21 => {   // dequantize_row_iq3_s: d qs[64] qh[8] signs[32] scales[4]
            let d = fp16(x, 0);
            let grid_table = iq_grid(4);
            let (mut qs, mut qh, mut signs) = (2, 66, 74);
            for ib32 in (0..QK_K / 32).step_by(2) {
                let db1 = d * f32::from(1 + 2 * (x[106 + ib32 / 2] & 0xf));
                let db2 = d * f32::from(1 + 2 * (x[106 + ib32 / 2] >> 4));
                for (db, h) in [(db1, 0), (db2, 1)] {
                    for l in 0..4 {
                        let grid1 = &grid_table[4 * (usize::from(x[qs + 2 * l]) | ((usize::from(x[qh + h]) << (8 - 2 * l)) & 256))..];
                        let grid2 = &grid_table[4 * (usize::from(x[qs + 2 * l + 1]) | ((usize::from(x[qh + h]) << (7 - 2 * l)) & 256))..];
                        let mut values = [0f32; 8];
                        for j in 0..4 {
                            values[j] = db * f32::from(grid1[j]) * sign(x[signs + l], j);
                            values[j + 4] = db * f32::from(grid2[j]) * sign(x[signs + l], j + 4);
                        }
                        y.extend(values);
                    }
                    qs += 8;
                    signs += 4;
                }
                qh += 2;
            }
        }
        19 => {   // dequantize_row_iq1_s: d qs[32] qh[8] (u16)
            let d = fp16(x, 0);
            let grid_table = iq_grid(5);
            let mut qs = 2;
            for ib in 0..QK_K / 32 {
                let qh = u16_at(x, 34 + 2 * ib);
                let dl = d * f32::from(2 * ((qh >> 12) & 7) + 1);
                let delta = if qh & 0x8000 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
                for l in 0..4 {
                    let index = usize::from(x[qs + l]) | (usize::from((qh >> (3 * l)) & 7) << 8);
                    // ggml's grid holds int8 -1, 0, 1; the stored bytes are those plus one.
                    let grid: Vec<i8> = grid_table[8 * index..8 * index + 8].iter().map(|g| *g as i8 - 1).collect();
                    y.extend((0..8).map(|j| dl * (f32::from(grid[j]) + delta)));
                }
                qs += 4;
            }
        }
        29 => {   // dequantize_row_iq1_m: qs[32] qh[16] scales[8]
            let sc = [u16_at(x, 48), u16_at(x, 50), u16_at(x, 52), u16_at(x, 54)];
            let scale = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
            let d = f16_to_f32(scale);
            let grid_table = iq_grid(5);
            let (mut qs, mut qh) = (0, 32);
            for ib in 0..QK_K / 32 {
                let dl1 = d * f32::from(2 * ((sc[ib / 2] >> (6 * (ib % 2))) & 0x7) + 1);
                let dl2 = d * f32::from(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
                let idx = [
                    usize::from(x[qs]) | ((usize::from(x[qh]) << 8) & 0x700),
                    usize::from(x[qs + 1]) | ((usize::from(x[qh]) << 4) & 0x700),
                    usize::from(x[qs + 2]) | ((usize::from(x[qh + 1]) << 8) & 0x700),
                    usize::from(x[qs + 3]) | ((usize::from(x[qh + 1]) << 4) & 0x700),
                ];
                let delta = [
                    if x[qh] & 0x08 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                    if x[qh] & 0x80 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                    if x[qh + 1] & 0x08 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                    if x[qh + 1] & 0x80 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                ];
                for l in 0..4 {
                    let grid: Vec<i8> = grid_table[8 * idx[l]..8 * idx[l] + 8].iter().map(|g| *g as i8 - 1).collect();
                    let dl = if l < 2 { dl1 } else { dl2 };
                    y.extend((0..8).map(|j| dl * (f32::from(grid[j]) + delta[l])));
                }
                qs += 4;
                qh += 2;
            }
        }
        23 => {   // dequantize_row_iq4_xs: d scales_h scales_l[4] qs[128]
            let d = fp16(x, 0);
            let scales_h = u16_at(x, 2);
            let mut qs = 8;
            for ib in 0..QK_K / 32 {
                let ls = i32::from(((x[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | ((((scales_h >> (2 * ib)) & 3) as u8) << 4));
                let dl = d * (ls - 32) as f32;
                let mut values = [0f32; 32];
                for j in 0..16 {
                    values[j] = dl * f32::from(KVALUES_IQ4NL[usize::from(x[qs + j] & 0xf)]);
                    values[j + 16] = dl * f32::from(KVALUES_IQ4NL[usize::from(x[qs + j] >> 4)]);
                }
                y.extend(values);
                qs += 16;
            }
        }
        42 => {   // Q2_0 (Straydog's own): d, then 64 two-bit codes LSB first; value d * (code - 1)
            let d = fp16(x, 0);
            y.extend((0..64).map(|i| d * f32::from(((x[2 + i / 4] >> (2 * (i % 4))) & 3) as i8 - 1)));
        }
        _ => panic!("no port of format {kind}"),
    }
    y
}

fn hex(text: &str) -> Vec<u8> {
    (0..text.len()).step_by(2).map(|at| u8::from_str_radix(&text[at..at + 2], 16).unwrap()).collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn decoded(kind: u32, data: &[u8]) -> Vec<f32> {
    let mut out = Vec::new();
    formats::decode_into(kind, data, &mut out).unwrap();
    out
}

fn ported(kind: u32, data: &[u8]) -> Vec<f32> {
    let (_, size) = formats::block(kind).unwrap();
    data.chunks_exact(size).flat_map(|block| ggml_dequantize(kind, block)).collect()
}

#[test]
fn real_blocks_decode_as_ggml_defines_them() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/format_blocks.json");
    let golden: Json = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut kinds = std::collections::BTreeSet::new();
    for case in golden["blocks"].as_array().unwrap() {
        let kind = case["kind"].as_u64().unwrap() as u32;
        let data = hex(case["data"].as_str().unwrap());
        let ours = decoded(kind, &data);
        assert_eq!(ours.len(), data.len() / formats::block(kind).unwrap().1 * formats::block(kind).unwrap().0);
        assert_eq!(bits(&ours), bits(&ported(kind, &data)), "{}", case["source"]);
        kinds.insert(kind);
    }
    assert_eq!(kinds.into_iter().collect::<Vec<_>>(), [0, 1, 2, 6, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 30, 42]);
}

/// xorshift64*: generated blocks are reproducible.
struct Random(u64);
impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// A finite FP16 of any magnitude, subnormals included.
    fn half(&mut self) -> [u8; 2] {
        let bits = (self.next() >> 48) as u16;
        let bits = if (bits >> 10) & 0x1f == 0x1f { bits & !0x4000 } else { bits };
        bits.to_le_bytes()
    }
}

#[test]
fn generated_blocks_of_every_format_decode_as_ggml_defines_them() {
    let mut random = Random(0x9e37_79b9_7f4a_7c15);
    // Byte offsets of each format's FP16 scales (kept finite; every other byte is arbitrary).
    let scales: &[(u32, &[usize])] = &[
        (1, &[0]), (2, &[0]), (6, &[0]), (8, &[0]), (20, &[0]), (39, &[]), (10, &[80, 82]), (11, &[108]),
        (12, &[0, 2]), (13, &[0, 2]), (14, &[208]), (16, &[0]), (17, &[0]), (18, &[0]), (19, &[0]), (21, &[0]),
        (22, &[0]), (23, &[0]), (29, &[]), (42, &[0]), (0, &[]), (30, &[]),
    ];
    for &(kind, halves) in scales {
        let (_, size) = formats::block(kind).unwrap();
        for _ in 0..2000 {
            let mut block: Vec<u8> = (0..size).map(|_| random.next() as u8).collect();
            for &at in halves {
                block[at..at + 2].copy_from_slice(&random.half());
            }
            if kind == 29 {
                // The scale's nibbles sit in the top of the four sub-scale words.
                let half = u16::from_le_bytes(random.half());
                for k in 0..4 {
                    block[49 + 2 * k] = (block[49 + 2 * k] & 0x0f) | (((half >> (4 * k)) & 15) << 4) as u8;
                }
            }
            if kind == 0 && !f32::from_le_bytes(block[..4].try_into().unwrap()).is_finite() {
                block[3] &= 0xbf;
            }
            if kind == 30 && (u16::from_le_bytes([block[0], block[1]]) >> 7) & 0xff == 0xff {
                block[1] &= 0xbf;
            }
            assert_eq!(bits(&decoded(kind, &block)), bits(&ported(kind, &block)), "kind {kind}: {block:02x?}");
        }
    }
}

#[test]
fn every_scale_exponent_of_mxfp4_decodes_as_ggml_defines_it() {
    let codes: Vec<u8> = (0..16).map(|i| (i * 17) as u8).collect();
    for e in 0..=255u8 {
        let block: Vec<u8> = std::iter::once(e).chain(codes.iter().copied()).collect();
        assert_eq!(bits(&decoded(39, &block)), bits(&ported(39, &block)), "e = {e}");
    }
}

#[test]
fn rows_of_a_tensor_decode_independently() {
    // A 3D tensor's rows are runs of whole blocks: any range decodes on its own.
    let mut random = Random(7);
    for kind in [10, 19, 39] {
        let (values, size) = formats::block(kind).unwrap();
        let mut data: Vec<u8> = (0..size * 40).map(|_| random.next() as u8).collect();
        for block in data.chunks_exact_mut(size) {
            match kind {
                10 => block[80..84].copy_from_slice(&[0x00, 0x2c, 0x00, 0x28]),
                19 => block[..2].copy_from_slice(&[0x00, 0x2c]),
                _ => block[0] = 120,
            }
        }
        let whole = decoded(kind, &data);
        let rows = 4;
        let row_bytes = data.len() / rows;
        for row in 0..rows {
            let part = decoded(kind, &data[row * row_bytes..(row + 1) * row_bytes]);
            assert_eq!(bits(&part), bits(&whole[row * (whole.len() / rows)..(row + 1) * (whole.len() / rows)]));
            assert_eq!(part.len(), row_bytes / size * values);
        }
    }
}

/// F32 weights that are all BF16 values (Unsloth's Qwen3.8 routers) go up as BF16: each value kept exactly, and anything else refused.
#[test]
fn f32_weights_of_bf16_values_narrow_exactly_and_others_are_refused() {
    let values: Vec<f32> = [0.0f32, -0.0, 1.0, -3.5, 1.0e-30, 6.0e30, f32::MIN_POSITIVE, 0.0078125].iter()
        .map(|value| f32::from_bits(value.to_bits() & 0xffff_0000)).collect();
    let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
    assert!(straydog_engine::gpu::bf16_exact(&bytes));
    let narrow = straydog_engine::gpu::narrow_bf16(&bytes, "router").unwrap();
    let widened: Vec<f32> = narrow.chunks_exact(2).map(|half| f32::from_bits(u32::from(u16::from_le_bytes([half[0], half[1]])) << 16)).collect();
    assert_eq!(bits(&widened), bits(&values));
    // (one value with a low bit set: not BF16, refused rather than rounded)
    let mut odd = bytes.clone();
    odd[8] = 1;
    assert!(!straydog_engine::gpu::bf16_exact(&odd));
    assert!(straydog_engine::gpu::narrow_bf16(&odd, "router").is_err());
}
