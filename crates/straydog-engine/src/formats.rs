//! GGML block geometry of the formats Straydog reads, exact decoding of the formats the
//! host decodes itself (embedding rows and small vectors), and the IQ lattice grids.
//!
//! Every decoded value is an FP16 or BF16 scale times small integers, exact in F32, so
//! the order of the products cannot change a bit.

/// kind -> (values per block, bytes per block).
pub fn block(kind: u32) -> Option<(usize, usize)> {
    Some(match kind {
        0 => (1, 4),
        1 | 30 => (1, 2),
        2 => (32, 18),
        6 => (32, 22),
        8 => (32, 34),
        20 => (32, 18),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        23 => (256, 136),
        16 => (256, 66),
        17 => (256, 74),
        18 => (256, 98),
        21 => (256, 110),
        22 => (256, 82),
        29 => (256, 56),
        42 => (64, 18),
        _ => return None,
    })
}

/// Formats the CPU expert engine evaluates (routed experts of a partially resident model).
pub const CPU_EXPERT_KINDS: [u32; 10] = [8, 16, 17, 18, 20, 21, 22, 23, 29, 42];

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

const IQ4_VALUES: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

/// Formats the host decodes: F32, F16, BF16, Q3_K, IQ4_NL, IQ3_S and IQ4_XS.
pub fn host_decodes(kind: u32) -> bool {
    matches!(kind, 0 | 1 | 30 | 11 | 20 | 21 | 23)
}

/// Decode whole blocks of `kind` into F32, appended to `out`.
pub fn decode_into(kind: u32, data: &[u8], out: &mut Vec<f32>) -> Result<(), String> {
    let (values, size) = block(kind).filter(|_| host_decodes(kind)).ok_or(format!("unsupported host format {kind}"))?;
    if !data.len().is_multiple_of(size) {
        return Err("incomplete block".into());
    }
    out.reserve(data.len() / size * values);
    for block in data.chunks_exact(size) {
        match kind {
            0 => out.push(f32::from_le_bytes(block.try_into().unwrap())),
            1 => out.push(half(block, 0)),
            30 => out.push(f32::from_bits(u32::from(u16::from_le_bytes([block[0], block[1]])) << 16)),
            11 => q3_k(block, out),
            20 => {
                let scale = half(block, 0);
                for shift in [0, 4] {
                    out.extend(block[2..18].iter().map(|byte| scale * f32::from(IQ4_VALUES[usize::from((byte >> shift) & 15)])));
                }
            }
            21 => iq3_s(block, out),
            23 => iq4_xs(block, out),
            _ => unreachable!(),
        }
    }
    Ok(())
}

/// Q3_K: hmask[32] qs[64] scales[12] d (f16); 16 groups of 16 with 6-bit signed scales.
fn q3_k(block: &[u8], out: &mut Vec<f32>) {
    let (hmask, qs, scales) = (&block[..32], &block[32..96], &block[96..108]);
    let d = half(block, 108);
    let scale = |group: usize| {
        let low = (scales[group % 8] >> (4 * (group / 8))) & 15;
        let high = (scales[8 + group % 4] >> (2 * (group / 4))) & 3;
        f32::from((low | (high << 4)) as i8 - 32)
    };
    // Value i of 256: in its 128-value half, quarter j (two bits of qs at shift 2j), lane l of 32.
    for i in 0..256 {
        let (half_index, within) = (i / 128, i % 128);
        let (quarter, lane) = (within / 32, within % 32);
        let low = (qs[half_index * 32 + lane] >> (2 * quarter)) & 3;
        let high = (hmask[lane] >> (half_index * 4 + quarter)) & 1;
        let q = low as i8 - if high == 0 { 4 } else { 0 };
        out.push(d * scale(i / 16) * f32::from(q));
    }
}

/// IQ3_S: d (f16) qs[64] qh[8] signs[32] scales[4]; 512x4 grid of odd values.
fn iq3_s(block: &[u8], out: &mut Vec<f32>) {
    let d = half(block, 0);
    let (qs, qh, signs, scales) = (&block[2..66], &block[66..74], &block[74..106], &block[106..110]);
    let grid = iq_grid(4);
    for i in 0..256 {
        let group = i / 32;   // eight groups of 32 values, a 4-bit scale each
        let scale = (scales[group / 2] >> (4 * (group % 2))) & 15;
        let db = d * f32::from(1 + 2 * scale);
        let index = i / 4;    // 64 grid entries of four values
        let high = u16::from((qh[index / 8] >> (index % 8)) & 1);
        let entry = usize::from(u16::from(qs[index]) | (high << 8));
        let sign = if (signs[i / 8] >> (i % 8)) & 1 == 0 { 1.0 } else { -1.0 };
        out.push(db * f32::from(grid[entry * 4 + i % 4]) * sign);
    }
}

/// IQ4_XS: d (f16) scales_h (u16) scales_l[4] qs[128]; eight groups of 32 with 6-bit scales.
fn iq4_xs(block: &[u8], out: &mut Vec<f32>) {
    let d = half(block, 0);
    let scales_h = u16::from_le_bytes([block[2], block[3]]);
    let (scales_l, qs) = (&block[4..8], &block[8..136]);
    for group in 0..8 {
        let low = (scales_l[group / 2] >> (4 * (group % 2))) & 15;
        let high = ((scales_h >> (2 * group)) & 3) as u8;
        let dl = d * f32::from((low | (high << 4)) as i8 - 32);
        let bytes = &qs[group * 16..group * 16 + 16];
        for shift in [0, 4] {
            out.extend(bytes.iter().map(|byte| dl * f32::from(IQ4_VALUES[usize::from((byte >> shift) & 15)])));
        }
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
