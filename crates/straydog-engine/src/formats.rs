//! GGML block geometry of the formats Straydog reads, exact decoding of every one of them to
//! F32 (the reference the GPU and CPU kernels are tested against, and the host's decoder of
//! embedding rows and small vectors), and the IQ lattice grids.
//!
//! Every decoded value is a scale times small integers, exact in F32, except that K-quants
//! with minimums (Q2_K, Q4_K, Q5_K) subtract two exact products and round once, as ggml's
//! definitions do; the order of the products cannot change a bit.

/// kind -> (values per block, bytes per block).
pub fn block(kind: u32) -> Option<(usize, usize)> {
    Some(match kind {
        0 => (1, 4),
        1 | 30 => (1, 2),
        2 => (32, 18),
        6 => (32, 22),
        8 => (32, 34),
        20 => (32, 18),
        39 => (32, 17),
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        23 => (256, 136),
        16 => (256, 66),
        17 => (256, 74),
        18 => (256, 98),
        19 => (256, 50),
        21 => (256, 110),
        22 => (256, 82),
        29 => (256, 56),
        42 => (64, 18),
        _ => return None,
    })
}

/// Formats the CPU expert engine evaluates (routed experts of a partially resident model).
pub const CPU_EXPERT_KINDS: [u32; 13] = [8, 10, 16, 17, 18, 19, 20, 21, 22, 23, 29, 39, 42];

/// Packed size of a `rows` x `cols` matrix of whole blocks.
pub fn matrix_bytes(kind: u32, rows: usize, cols: usize) -> Option<usize> {
    let (values, size) = block(kind)?;
    (cols.is_multiple_of(values)).then(|| rows * cols / values * size)
}

/// IQ2_XXS (256x8), IQ2_XS (512x8), IQ2_S (1024x8), IQ3_XXS (256x4), IQ3_S (512x4) and
/// IQ1_S/IQ1_M (2048x8) grids, one byte per value (IQ1 values -1, 0, 1 stored as 0, 1, 2).
/// From ggml's format definitions.
pub const IQ_GRIDS: &[u8; 33792] = include_bytes!("iq_grids.bin");
const IQ_GRID_OFFSETS: [usize; 6] = [0, 2048, 6144, 14336, 15360, 17408];

/// One grid of [`IQ_GRIDS`] by its position in the list above.
pub fn iq_grid(index: usize) -> &'static [u8] {
    let end = IQ_GRID_OFFSETS.get(index + 1).copied().unwrap_or(IQ_GRIDS.len());
    &IQ_GRIDS[IQ_GRID_OFFSETS[index]..end]
}

pub fn f16_to_f32(half: u16) -> f32 {
    let sign = u32::from(half >> 15) << 31;
    let exponent = u32::from((half >> 10) & 0x1f);
    let mantissa = u32::from(half & 0x3ff);
    let bits = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            // Subnormal: shift the mantissa up to an implicit leading one.
            let (mut exponent, mut mantissa) = (127 - 15 + 1, mantissa);
            while mantissa & 0x400 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            sign | (exponent << 23) | ((mantissa & 0x3ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(bits)
}

fn half(data: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([data[at], data[at + 1]]))
}

fn word(data: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(data[at..at + 4].try_into().unwrap())
}

const IQ4_VALUES: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
/// E2M1 values doubled (MXFP4); the block scale is halved to match.
const FP4_VALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// Formats the host decodes: every format of [`block`].
pub fn host_decodes(kind: u32) -> bool {
    block(kind).is_some()
}

/// Decode whole blocks of `kind` into F32, appended to `out`. Blocks are independent: any
/// range of whole blocks (a row, rows of a 2D or 3D tensor) decodes on its own.
pub fn decode_into(kind: u32, data: &[u8], out: &mut Vec<f32>) -> Result<(), String> {
    let (values, size) = block(kind).filter(|_| host_decodes(kind)).ok_or(format!("unsupported host format {kind}"))?;
    if !data.len().is_multiple_of(size) {
        return Err("incomplete block".into());
    }
    if data.is_empty() {
        return Ok(());
    }
    let start = out.len();
    out.resize(start + data.len() / size * values, 0.0);
    let target = &mut out[start..];
    // Each format's loop is its own instance, its block decoder inlined.
    fn blocks(data: &[u8], target: &mut [f32], size: usize, decode: impl Fn(&[u8], &mut [f32])) {
        let values = target.len() / (data.len() / size);
        data.chunks_exact(size).zip(target.chunks_exact_mut(values)).for_each(|(block, out)| decode(block, out));
    }
    match kind {
        0 => target.iter_mut().zip(data.chunks_exact(4)).for_each(|(o, b)| *o = f32::from_le_bytes(b.try_into().unwrap())),
        1 => target.iter_mut().zip(data.chunks_exact(2)).for_each(|(o, b)| *o = half(b, 0)),
        30 => target.iter_mut().zip(data.chunks_exact(2))
            .for_each(|(o, b)| *o = f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)),
        2 => blocks(data, target, size, q4_0),
        6 => blocks(data, target, size, q5_0),
        8 => blocks(data, target, size, q8_0),
        20 => blocks(data, target, size, iq4_nl),
        39 => blocks(data, target, size, mxfp4),
        10 => blocks(data, target, size, q2_k),
        11 => blocks(data, target, size, q3_k),
        12 => blocks(data, target, size, k_quant_45::<false>),
        13 => blocks(data, target, size, k_quant_45::<true>),
        14 => blocks(data, target, size, q6_k),
        16 => blocks(data, target, size, iq2_xxs),
        17 => blocks(data, target, size, iq2_xs),
        18 => blocks(data, target, size, iq3_xxs),
        19 => blocks(data, target, size, iq1_s),
        21 => blocks(data, target, size, iq3_s),
        22 => blocks(data, target, size, iq2_s),
        23 => blocks(data, target, size, iq4_xs),
        29 => blocks(data, target, size, iq1_m),
        42 => blocks(data, target, size, q2_0),
        _ => unreachable!(),
    }
    Ok(())
}

/// Sixteen bytes of nibbles: low nibbles are values 0-15, high nibbles 16-31, each mapped by `value`.
fn nibbles(codes: &[u8], out: &mut [f32], value: impl Fn(u8, usize) -> f32) {
    let (low, high) = out.split_at_mut(16);
    for j in 0..16 {
        low[j] = value(codes[j] & 15, j);
        high[j] = value(codes[j] >> 4, j + 16);
    }
}

/// Q4_0: d (f16), then 16 bytes of four-bit values minus 8.
fn q4_0(block: &[u8], out: &mut [f32]) {
    let d = half(block, 0);
    nibbles(&block[2..18], out, |nibble, _| d * f32::from(nibble as i8 - 8));
}

/// Q5_0: d (f16), 32 high bits (bit i of value i), then 16 bytes of low nibbles; values minus 16.
fn q5_0(block: &[u8], out: &mut [f32]) {
    let (d, high) = (half(block, 0), word(block, 2));
    nibbles(&block[6..22], out, |nibble, i| d * f32::from((nibble | (((high >> i) & 1) as u8) << 4) as i8 - 16));
}

/// IQ4_NL: d (f16), then 16 bytes of nonlinear four-bit values.
fn iq4_nl(block: &[u8], out: &mut [f32]) {
    let d = half(block, 0);
    nibbles(&block[2..18], out, |nibble, _| d * f32::from(IQ4_VALUES[usize::from(nibble)]));
}

/// Q8_0: d (f16), then 32 signed bytes.
fn q8_0(block: &[u8], out: &mut [f32]) {
    let d = half(block, 0);
    out.iter_mut().zip(&block[2..34]).for_each(|(o, q)| *o = d * f32::from(*q as i8));
}

/// Q2_0 (Straydog's own): d (f16), then 64 two-bit codes, LSB first; a value is d * (code - 1).
fn q2_0(block: &[u8], out: &mut [f32]) {
    let d = half(block, 0);
    for (i, o) in out.iter_mut().enumerate() {
        *o = d * f32::from(((block[2 + i / 4] >> (2 * (i % 4))) & 3) as i8 - 1);
    }
}

/// MXFP4: an E8M0 exponent e, then 16 bytes of E2M1 codes. The doubled values take the scale
/// 2^(e - 128), a power of two (subnormal below e = 2).
fn mxfp4(block: &[u8], out: &mut [f32]) {
    let e = u32::from(block[0]);
    let scale = f32::from_bits(if e < 2 { 0x0020_0000 << e } else { (e - 1) << 23 });
    nibbles(&block[1..17], out, |nibble, _| f32::from(FP4_VALUES[usize::from(nibble)]) * scale);
}

/// Q2_K: scales[16] qs[64] d dmin (f16); 16 groups of 16 with a 4-bit scale and minimum each.
/// In each 128-value half, quarter j takes two bits of qs at shift 2j, lanes 0-15 and 16-31
/// with a scale of their own.
fn q2_k(block: &[u8], out: &mut [f32]) {
    let (scales, qs) = (&block[..16], &block[16..80]);
    let (d, minimum) = (half(block, 80), half(block, 82));
    for (group, values) in out.chunks_exact_mut(16).enumerate() {
        let (half_index, quarter, side) = (group / 8, (group / 2) % 4, group % 2);
        let scale = scales[8 * half_index + 2 * quarter + side];
        let (dl, ml) = (d * f32::from(scale & 15), minimum * f32::from(scale >> 4));
        let codes = &qs[32 * half_index + 16 * side..][..16];
        values.iter_mut().zip(codes).for_each(|(o, q)| *o = dl * f32::from((q >> (2 * quarter)) & 3) - ml);
    }
}

/// Q3_K: hmask[32] qs[64] scales[12] d (f16); 16 groups of 16 with 6-bit signed scales.
fn q3_k(block: &[u8], out: &mut [f32]) {
    let (hmask, qs, scales) = (&block[..32], &block[32..96], &block[96..108]);
    let d = half(block, 108);
    // Value i of 256: in its 128-value half, quarter j (two bits of qs at shift 2j), lane l of 32.
    for (group, values) in out.chunks_exact_mut(16).enumerate() {
        let low = (scales[group % 8] >> (4 * (group / 8))) & 15;
        let high = (scales[8 + group % 4] >> (2 * (group / 4))) & 3;
        let scale = f32::from((low | (high << 4)) as i8 - 32);
        let (half_index, quarter, side) = (group / 8, (group / 2) % 4, group % 2);
        for (l, o) in values.iter_mut().enumerate() {
            let lane = 16 * side + l;
            let low = (qs[half_index * 32 + lane] >> (2 * quarter)) & 3;
            let high = (hmask[lane] >> (half_index * 4 + quarter)) & 1;
            *o = d * scale * f32::from(low as i8 - if high == 0 { 4 } else { 0 });
        }
    }
}

/// Q4_K and Q5_K: d dmin (f16) scales[12] (Q5_K: qh[32]) qs[128]; eight groups of 32 with
/// 6-bit scales and minimums. Group g's codes are nibble g % 2 of bytes 32 * (g / 2) ..
fn k_quant_45<const FIFTH: bool>(block: &[u8], out: &mut [f32]) {
    let (d, minimum, scales) = (half(block, 0), half(block, 2), &block[4..16]);
    let qs = &block[if FIFTH { 48 } else { 16 }..];
    for (group, values) in out.chunks_exact_mut(32).enumerate() {
        let (scale, min) = if group < 4 {
            (scales[group] & 63, scales[group + 4] & 63)
        } else {
            ((scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4), (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4))
        };
        let (dl, ml) = (d * f32::from(scale), minimum * f32::from(min));
        let codes = &qs[32 * (group / 2)..][..32];
        for (lane, o) in values.iter_mut().enumerate() {
            let mut code = (codes[lane] >> (4 * (group % 2))) & 15;
            if FIFTH {
                code |= ((block[16 + lane] >> group) & 1) << 4;
            }
            *o = dl * f32::from(code) - ml;
        }
    }
}

/// Q6_K: ql[128] qh[64] scales[16] (i8) d (f16); 16 groups of 16 six-bit values minus 32.
/// In each 128-value half, quarter j takes nibble j / 2 of ql bytes 32 * (j % 2) .. and two
/// bits of qh at shift 2j.
fn q6_k(block: &[u8], out: &mut [f32]) {
    let (ql, qh, scales) = (&block[..128], &block[128..192], &block[192..208]);
    let d = half(block, 208);
    for (group, values) in out.chunks_exact_mut(16).enumerate() {
        let (half_index, quarter, side) = (group / 8, (group / 2) % 4, group % 2);
        let scale = f32::from(scales[group] as i8);
        let low = &ql[64 * half_index + 32 * (quarter % 2) + 16 * side..][..16];
        let high = &qh[32 * half_index + 16 * side..][..16];
        for ((o, l), h) in values.iter_mut().zip(low).zip(high) {
            let code = ((l >> (4 * (quarter / 2))) & 15) | (((h >> (2 * quarter)) & 3) << 4);
            *o = d * scale * f32::from(code as i8 - 32);
        }
    }
}

/// Seven stored sign bits plus the eighth, which keeps the number of negatives even.
fn iq_signs(bits: u32) -> u32 {
    bits | ((bits.count_ones() & 1) << 7)
}

/// Eight values scale * grid value, negated where `signs` has a bit (the products are exact).
fn iq_signed8(out: &mut [f32], scale: f32, grid: &[u8], signs: u32) {
    for (j, (o, g)) in out.iter_mut().zip(grid).enumerate() {
        *o = scale * f32::from(*g) * if (signs >> j) & 1 == 0 { 1.0 } else { -1.0 };
    }
}

/// IQ2_XXS: d (f16), then per 32 values four grid bytes and a word of 4x7 sign bits and a scale.
fn iq2_xxs(block: &[u8], out: &mut [f32]) {
    let d = half(block, 0);
    let grid = iq_grid(0);
    for (ib, values) in out.chunks_exact_mut(32).enumerate() {
        let aux = word(block, 6 + 8 * ib);
        let db = d * (0.5 + (aux >> 28) as f32) * 0.25;
        for (l, eight) in values.chunks_exact_mut(8).enumerate() {
            let row = usize::from(block[2 + 8 * ib + l]);
            iq_signed8(eight, db, &grid[8 * row..8 * row + 8], iq_signs((aux >> (7 * l)) & 127));
        }
    }
}

/// IQ2_XS: d (f16) qs[32] (u16: 9-bit grid row, 7 sign bits) scales[8] (two nibbles per 32 values).
fn iq2_xs(block: &[u8], out: &mut [f32]) {
    let d = half(block, 0);
    let grid = iq_grid(1);
    for (ib, values) in out.chunks_exact_mut(32).enumerate() {
        for (l, eight) in values.chunks_exact_mut(8).enumerate() {
            let q = u32::from(u16::from_le_bytes([block[2 + 2 * (4 * ib + l)], block[3 + 2 * (4 * ib + l)]]));
            let db = d * (0.5 + f32::from((block[66 + ib] >> (4 * (l / 2))) & 15)) * 0.25;
            let row = (q & 511) as usize;
            iq_signed8(eight, db, &grid[8 * row..8 * row + 8], iq_signs(q >> 9));
        }
    }
}

/// IQ2_S: d (f16) qs[32] signs[32] qh[8] (two high bits per grid row) scales[8].
fn iq2_s(block: &[u8], out: &mut [f32]) {
    let d = half(block, 0);
    let grid = iq_grid(2);
    for (ib, values) in out.chunks_exact_mut(32).enumerate() {
        for (l, eight) in values.chunks_exact_mut(8).enumerate() {
            let row = usize::from(block[2 + 4 * ib + l]) | ((usize::from(block[66 + ib]) << (8 - 2 * l)) & 0x300);
            let db = d * (0.5 + f32::from((block[74 + ib] >> (4 * (l / 2))) & 15)) * 0.25;
            iq_signed8(eight, db, &grid[8 * row..8 * row + 8], u32::from(block[34 + 4 * ib + l]));
        }
    }
}

/// IQ3_XXS: d (f16) qs[64] (grid rows of four) and per 32 values a word of 4x7 sign bits and a scale.
fn iq3_xxs(block: &[u8], out: &mut [f32]) {
    let d = half(block, 0);
    let grid = iq_grid(3);
    for (ib, values) in out.chunks_exact_mut(32).enumerate() {
        let aux = word(block, 66 + 4 * ib);
        let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
        for (l, eight) in values.chunks_exact_mut(8).enumerate() {
            let (first, second) = (usize::from(block[2 + 8 * ib + 2 * l]), usize::from(block[3 + 8 * ib + 2 * l]));
            let rows: [u8; 8] = std::array::from_fn(|j| if j < 4 { grid[4 * first + j] } else { grid[4 * second + j - 4] });
            iq_signed8(eight, db, &rows, iq_signs((aux >> (7 * l)) & 127));
        }
    }
}

/// IQ3_S: d (f16) qs[64] qh[8] signs[32] scales[4]; 512x4 grid of odd values.
fn iq3_s(block: &[u8], out: &mut [f32]) {
    let d = half(block, 0);
    let grid = iq_grid(4);
    for (ib, values) in out.chunks_exact_mut(32).enumerate() {
        // Eight groups of 32 values, a 4-bit scale each; 64 grid entries of four values.
        let db = d * f32::from(1 + 2 * ((block[106 + ib / 2] >> (4 * (ib % 2))) & 15));
        for (l, eight) in values.chunks_exact_mut(8).enumerate() {
            let rows: [u8; 8] = std::array::from_fn(|j| {
                let index = 8 * ib + 2 * l + j / 4;
                let high = usize::from((block[66 + index / 8] >> (index % 8)) & 1);
                grid[4 * (usize::from(block[2 + index]) | (high << 8)) + j % 4]
            });
            iq_signed8(eight, db, &rows, u32::from(block[74 + 4 * ib + l]));
        }
    }
}

/// IQ1_S: d (f16) qs[32] qh[8] (u16: four 3-bit high parts of grid rows, a 3-bit scale and the
/// delta's sign per 32 values). A value is d * (2 * scale + 1) * (grid value + delta).
fn iq1_s(block: &[u8], out: &mut [f32]) {
    let d = half(block, 0);
    let grid = iq_grid(5);
    for (ib, values) in out.chunks_exact_mut(32).enumerate() {
        let qh = u16::from_le_bytes([block[34 + 2 * ib], block[35 + 2 * ib]]);
        let dl = d * f32::from(2 * ((qh >> 12) & 7) + 1);
        let delta = if qh & 0x8000 != 0 { -0.125 } else { 0.125 };
        for (l, eight) in values.chunks_exact_mut(8).enumerate() {
            let row = usize::from(block[2 + 4 * ib + l]) | (usize::from((qh >> (3 * l)) & 7) << 8);
            eight.iter_mut().zip(&grid[8 * row..8 * row + 8]).for_each(|(o, g)| *o = dl * (f32::from(*g) - 1.0 + delta));
        }
    }
}

/// IQ1_M: qs[32] qh[16] (per grid row three high bits and the delta's sign) and four sub-scale
/// words (3 bits per 16 values) whose top nibbles hold the FP16 scale.
fn iq1_m(block: &[u8], out: &mut [f32]) {
    let words: [u16; 4] = std::array::from_fn(|k| u16::from_le_bytes([block[48 + 2 * k], block[49 + 2 * k]]));
    let bits = (words[0] >> 12) | ((words[1] >> 8) & 0xf0) | ((words[2] >> 4) & 0xf00) | (words[3] & 0xf000);
    let d = f16_to_f32(bits);
    let grid = iq_grid(5);
    for (ib, values) in out.chunks_exact_mut(32).enumerate() {
        for (l, eight) in values.chunks_exact_mut(8).enumerate() {
            let dl = d * f32::from(2 * ((words[ib / 2] >> (6 * (ib % 2) + 3 * (l / 2))) & 7) + 1);
            let high = block[32 + 2 * ib + l / 2] >> (4 * (l % 2));
            let delta = if high & 8 != 0 { -0.125 } else { 0.125 };
            let row = usize::from(block[4 * ib + l]) | (usize::from(high & 7) << 8);
            eight.iter_mut().zip(&grid[8 * row..8 * row + 8]).for_each(|(o, g)| *o = dl * (f32::from(*g) - 1.0 + delta));
        }
    }
}

/// IQ4_XS: d (f16) scales_h (u16) scales_l[4] qs[128]; eight groups of 32 with 6-bit scales.
fn iq4_xs(block: &[u8], out: &mut [f32]) {
    let d = half(block, 0);
    let scales_h = u16::from_le_bytes([block[2], block[3]]);
    let (scales_l, qs) = (&block[4..8], &block[8..136]);
    for (group, values) in out.chunks_exact_mut(32).enumerate() {
        let low = (scales_l[group / 2] >> (4 * (group % 2))) & 15;
        let high = ((scales_h >> (2 * group)) & 3) as u8;
        let dl = d * f32::from((low | (high << 4)) as i8 - 32);
        nibbles(&qs[group * 16..group * 16 + 16], values, |nibble, _| dl * f32::from(IQ4_VALUES[usize::from(nibble)]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_precision_converts_exactly() {
        for bits in 0..=u16::MAX {
            let ours = f16_to_f32(bits);
            let exponent = (bits >> 10) & 0x1f;
            if exponent == 0x1f && bits & 0x3ff != 0 {
                assert!(ours.is_nan());
                continue;
            }
            // Every half is exactly representable: converting back by rounding must recover it.
            let mantissa = f64::from(bits & 0x3ff);
            let magnitude = if exponent == 0 {
                mantissa * 2f64.powi(-24)
            } else if exponent == 0x1f {
                f64::INFINITY
            } else {
                (1.0 + mantissa / 1024.0) * 2f64.powi(i32::from(exponent) - 15)
            };
            let expected = if bits >> 15 == 1 { -magnitude } else { magnitude };
            assert_eq!(f64::from(ours), expected, "{bits:#06x}");
        }
    }
}
