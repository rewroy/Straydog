//! Images for DeepSeek-V4.1's vision encoder, as the reference's `inference/image_processor.py`
//! prepares them: decoded to 8-bit RGB, sized by its plan (at least `min_pixels`, at most
//! `max_tokens` model tokens), padded to that size with gray as Pillow's `ImageOps.pad` does (its
//! bicubic resampling, bit for bit), and cut into 14x14 patches whose values are BF16.
//!
//! The model reads an image as a span of tokens: `[START] ([IMAGE] * columns [NEWLINE]) * rows
//! [END]`, the IMAGE slots taking the encoder's rows in reading order.

use image::{ImageDecoder, ImageFormat, ImageReader};

/// The reference's vision settings (its `config.json`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionConfig {
    pub patch: usize,
    /// The aligner's downsampling: `downsample` x `downsample` patches make a model token.
    pub downsample: usize,
    /// Most model tokens an image's span takes.
    pub max_tokens: usize,
    /// A smaller image is scaled up to about this many pixels first.
    pub min_pixels: usize,
}

impl VisionConfig {
    /// DeepSeek-V4.1-Flash's.
    pub const DEEPSEEK41: VisionConfig = VisionConfig { patch: 14, downsample: 3, max_tokens: 1024, min_pixels: 544 * 544 };
}

/// Token ids from this one up are image positions. Each image span's are its own (from its image's
/// digest), so a session tells sequences with different images apart by their ids alone.
pub const IMAGE_TOKEN: u32 = 1 << 31;

/// What a span's position holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slot {
    Start,
    /// The encoder's row this position takes.
    Image(usize),
    NewLine,
    End,
}

/// The positions of an image's span: one row of `columns` IMAGE slots and a NEWLINE for each
/// of `rows`, between START and END.
pub fn span(rows: usize, columns: usize) -> Vec<Slot> {
    let mut slots = Vec::with_capacity(span_tokens(rows, columns));
    slots.push(Slot::Start);
    for row in 0..rows {
        slots.extend((0..columns).map(|column| Slot::Image(row * columns + column)));
        slots.push(Slot::NewLine);
    }
    slots.push(Slot::End);
    slots
}

pub fn span_tokens(rows: usize, columns: usize) -> usize {
    rows * (columns + 1) + 2
}

/// 8-bit RGB, three bytes a pixel, rows top first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rgb {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

/// Largest image a request may give, in pixels and in encoded bytes.
pub const MAX_PIXELS: u64 = 64 << 20;
pub const MAX_BYTES: usize = 32 << 20;

/// Decode a PNG, JPEG, WebP or GIF (its first frame) to RGB, as Pillow's `convert("RGB")` takes
/// it: an alpha channel dropped, gray repeated, no EXIF rotation.
pub fn decode(bytes: &[u8]) -> Result<Rgb, String> {
    if bytes.len() > MAX_BYTES {
        return Err(format!("an image of more than {} MB", MAX_BYTES >> 20));
    }
    let reader = ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format().map_err(|exc| exc.to_string())?;
    match reader.format() {
        Some(ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::WebP | ImageFormat::Gif) => {}
        _ => return Err("an image that is not PNG, JPEG, WebP or GIF".into()),
    }
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(MAX_PIXELS * 8);
    let mut reader = reader;
    reader.limits(limits);
    let decoder = reader.into_decoder().map_err(|exc| format!("the image does not decode: {exc}"))?;
    let (width, height) = decoder.dimensions();
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > MAX_PIXELS {
        return Err(format!("an image of {width}x{height} pixels (at most {} million)", MAX_PIXELS >> 20));
    }
    let image = image::DynamicImage::from_decoder(decoder).map_err(|exc| format!("the image does not decode: {exc}"))?;
    let rgb = image.into_rgb8();
    Ok(Rgb { width: rgb.width() as usize, height: rgb.height() as usize, pixels: rgb.into_raw() })
}

/// The reference's size for an image: its model token grid and its pixel size (a multiple of
/// the patch both ways).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plan {
    pub rows: usize,
    pub columns: usize,
    pub height: usize,
    pub width: usize,
}

impl Plan {
    /// The encoder's patch grid (rows, columns).
    pub fn patches(&self, config: &VisionConfig) -> (usize, usize) {
        (self.height / config.patch, self.width / config.patch)
    }

    pub fn tokens(&self) -> usize {
        span_tokens(self.rows, self.columns)
    }
}

/// `plan_image_grid` (no aspect limit: the model's is none).
pub fn plan(width: usize, height: usize, config: &VisionConfig) -> Plan {
    let p = config.patch;
    let (mut width, mut height) = (width, height);
    if width * height > 0 && width * height < config.min_pixels {
        let ratio = (config.min_pixels as f64 / (width * height) as f64).sqrt();
        width = (width as f64 * ratio) as usize;
        height = (height as f64 * ratio) as usize;
    }
    let best_width = (width as f64 / p as f64).ceil() as usize * p;
    let best_height = (height as f64 / p as f64).ceil() as usize * p;
    let llm = |height: usize, width: usize| ((height / p).div_ceil(config.downsample), (width / p).div_ceil(config.downsample));
    let (mut rows, mut columns) = llm(best_height, best_width);
    let (mut best_height, mut best_width) = (best_height, best_width);
    if span_tokens(rows, columns) > config.max_tokens {
        (best_height, best_width) = solve_resize(height, width, config);
        (rows, columns) = llm(best_height, best_width);
    }
    Plan { rows, columns, height: best_height, width: best_width }
}

/// `solve_resize_ratio`: the largest aspect-keeping size whose token grid fits.
fn solve_resize(height: usize, width: usize, config: &VisionConfig) -> (usize, usize) {
    let (p, n) = (config.patch, config.max_tokens);
    let r = height as f64 / width as f64;
    let max_w = ((n - 2) as f64 / r + 0.25).sqrt() - 0.5;
    let max_h = max_w * r;
    let cell = p * config.downsample;
    if max_w < 1.0 {
        return ((n - 2) / 2 * cell, cell);
    }
    if max_h < 1.0 {
        return (cell, (n - 3) * cell);
    }
    let beta = f64::min((max_w.floor() as usize * cell) as f64 / width as f64, (max_h.floor() as usize * cell) as f64 / height as f64);
    (((height as f64 * beta) / p as f64).floor() as usize * p, ((width as f64 * beta) / p as f64).floor() as usize * p)
}

/// Pillow's fixed point for 8-bit resampling coefficients.
const PRECISION_BITS: u32 = 32 - 8 - 2;

fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Pillow's `precompute_coeffs` for its bicubic filter (support 2) and `normalize_coeffs_8bpc`:
/// each output's first input, input count and fixed-point weights (`size` apart).
fn coefficients(input: usize, output: usize) -> (Vec<(usize, usize)>, Vec<i32>, usize) {
    let scale = input as f64 / output as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let size = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(output);
    let mut weights = vec![0i32; output * size];
    let mut k = vec![0f64; size];
    for xx in 0..output {
        let center = (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = ((center + support + 0.5) as i64).min(input as i64) as usize - xmin;
        let mut ww = 0.0;
        k.fill(0.0);
        for (x, value) in k.iter_mut().enumerate().take(xmax) {
            *value = bicubic((x as f64 + xmin as f64 - center + 0.5) * ss);
            ww += *value;
        }
        for (x, value) in k.iter().enumerate() {
            let w = if x < xmax && ww != 0.0 { value / ww } else { *value };
            weights[xx * size + x] = if w < 0.0 { (-0.5 + w * f64::from(1u32 << PRECISION_BITS)) as i32 }
                                     else { (0.5 + w * f64::from(1u32 << PRECISION_BITS)) as i32 };
        }
        bounds.push((xmin, xmax));
    }
    (bounds, weights, size)
}

fn clip8(value: i32) -> u8 {
    (value >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Pillow's `Image.resize(size, BICUBIC)` of 8-bit RGB: horizontal then vertical, each pass's
/// values rounded to 8 bits, the horizontal one over only the rows the vertical one reads.
pub fn resize(image: &Rgb, width: usize, height: usize) -> Rgb {
    if (image.width, image.height) == (width, height) {
        return image.clone();
    }
    let (horizontal, vertical) = (width != image.width, height != image.height);
    let (h_bounds, h_weights, h_size) = coefficients(image.width, width);
    let (mut v_bounds, v_weights, v_size) = coefficients(image.height, height);
    let first = v_bounds[0].0;
    let last = v_bounds[height - 1].0 + v_bounds[height - 1].1;
    let mut current = image.clone();
    if horizontal {
        for bound in &mut v_bounds {
            bound.0 -= first;
        }
        let rows = last - first;
        let mut pixels = vec![0u8; rows * width * 3];
        for yy in 0..rows {
            let source = &image.pixels[(yy + first) * image.width * 3..(yy + first + 1) * image.width * 3];
            for (xx, &(xmin, count)) in h_bounds.iter().enumerate() {
                let k = &h_weights[xx * h_size..xx * h_size + count];
                let mut sums = [1i32 << (PRECISION_BITS - 1); 3];
                for (x, &weight) in k.iter().enumerate() {
                    for (channel, sum) in sums.iter_mut().enumerate() {
                        *sum += i32::from(source[(x + xmin) * 3 + channel]) * weight;
                    }
                }
                for (channel, sum) in sums.into_iter().enumerate() {
                    pixels[(yy * width + xx) * 3 + channel] = clip8(sum);
                }
            }
        }
        current = Rgb { width, height: rows, pixels };
    }
    if vertical {
        let mut pixels = vec![0u8; height * current.width * 3];
        for (yy, &(ymin, count)) in v_bounds.iter().enumerate() {
            let k = &v_weights[yy * v_size..yy * v_size + count];
            for xx in 0..current.width {
                let mut sums = [1i32 << (PRECISION_BITS - 1); 3];
                for (y, &weight) in k.iter().enumerate() {
                    for (channel, sum) in sums.iter_mut().enumerate() {
                        *sum += i32::from(current.pixels[((y + ymin) * current.width + xx) * 3 + channel]) * weight;
                    }
                }
                for (channel, sum) in sums.into_iter().enumerate() {
                    pixels[(yy * current.width + xx) * 3 + channel] = clip8(sum);
                }
            }
        }
        current = Rgb { width: current.width, height, pixels };
    }
    current
}

/// Pillow's `ImageOps.pad(image, (width, height), BICUBIC, color)`: the image scaled to fit,
/// centered (its offset rounded half to even, as Python rounds) on that color.
pub fn pad(image: &Rgb, width: usize, height: usize, color: [u8; 3]) -> Rgb {
    let ratio = image.width as f64 / image.height as f64;
    let target = width as f64 / height as f64;
    let (mut fit_width, mut fit_height) = (width, height);
    if ratio != target {
        if ratio > target {
            fit_height = (image.height as f64 / image.width as f64 * width as f64).round_ties_even() as usize;
        } else {
            fit_width = (image.width as f64 / image.height as f64 * height as f64).round_ties_even() as usize;
        }
    }
    let resized = resize(image, fit_width, fit_height);
    if (fit_width, fit_height) == (width, height) {
        return resized;
    }
    let mut pixels: Vec<u8> = color.iter().copied().cycle().take(width * height * 3).collect();
    let (x, y) = if fit_width != width {
        (((width - fit_width) as f64 * 0.5).round_ties_even() as usize, 0)
    } else {
        (0, ((height - fit_height) as f64 * 0.5).round_ties_even() as usize)
    };
    for row in 0..fit_height {
        let at = ((row + y) * width + x) * 3;
        pixels[at..at + fit_width * 3].copy_from_slice(&resized.pixels[row * fit_width * 3..(row + 1) * fit_width * 3]);
    }
    Rgb { width, height, pixels }
}

/// An image ready for the encoder: its plan and its patches, `patch * patch` values of each
/// channel a patch, the patches in reading order, each value BF16 (its bits).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Patches {
    pub plan: Plan,
    pub rows: usize,
    pub columns: usize,
    pub values: Vec<u16>,
}

/// F32 to BF16 bits, rounded to nearest even (as torch's `.to(bfloat16)`).
pub fn bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) | 0x40) as u16;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// `load_image`: plan, pad with gray (127), scale each value to [-1, 1] in F32 and round it to BF16.
pub fn patches(image: &Rgb, config: &VisionConfig) -> Patches {
    let plan = plan(image.width, image.height, config);
    let padded = pad(image, plan.width, plan.height, [127; 3]);
    let p = config.patch;
    let (rows, columns) = plan.patches(config);
    let mut values = Vec::with_capacity(rows * columns * 3 * p * p);
    for row in 0..rows {
        for column in 0..columns {
            for channel in 0..3 {
                for y in 0..p {
                    for x in 0..p {
                        let pixel = padded.pixels[((row * p + y) * plan.width + column * p + x) * 3 + channel];
                        values.push(bf16((f32::from(pixel) / 255.0 - 0.5) / 0.5));
                    }
                }
            }
        }
    }
    Patches { plan, rows, columns, values }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: VisionConfig = VisionConfig::DEEPSEEK41;

    #[test]
    fn plans_match_the_reference() {
        // (width, height) -> (rows, columns, height, width), from image_processor.plan_image_grid
        let cases = [
            ((640, 480), (12, 16, 490, 644)),
            ((800, 600), (15, 20, 602, 812)),
            ((1280, 853), (21, 31, 854, 1288)),
            ((4000, 3000), (27, 36, 1134, 1512)),
            ((100, 5000), (120, 3, 5012, 112)),
            ((5000, 100), (3, 120, 112, 5012)),
            ((544, 544), (13, 13, 546, 546)),
            ((1, 1), (13, 13, 546, 546)),
            ((17, 3000), (173, 1, 7238, 42)),
            ((3001, 7), (1, 269, 28, 11270)),
            ((1920, 1080), (23, 41, 966, 1708)),
            ((300, 200), (11, 16, 448, 672)),
        ];
        for ((width, height), (rows, columns, h, w)) in cases {
            let plan = plan(width, height, &CONFIG);
            assert_eq!((plan.rows, plan.columns, plan.height, plan.width), (rows, columns, h, w), "{width}x{height}");
            assert!(plan.tokens() <= CONFIG.max_tokens);
        }
    }

    /// Pseudo-random RGB: x <- (x * 1103515245 + 12345) mod 2^31, each byte (x >> 16) & 255.
    fn noise(width: usize, height: usize, seed: u64) -> Rgb {
        let mut x = seed;
        let pixels = (0..width * height * 3).map(|_| {
            x = (x * 1_103_515_245 + 12_345) % (1 << 31);
            ((x >> 16) & 255) as u8
        }).collect();
        Rgb { width, height, pixels }
    }

    #[test]
    fn padding_matches_pillow() {
        use sha2::{Digest, Sha256};
        // ImageOps.pad(image, plan size, color=(127, 127, 127)) of noise(width, height, seed), Pillow 12.2
        let cases = [
            ((37, 23, 1), "8bb184e41eee932e95209ba538c2cad419da44fd621f9f03eb9d2b1cc63965bf"),
            ((300, 200, 2), "7f8e398af2715a512ba5a07491deecb69ec302c4b88465f02052074cecbbf9ac"),
            ((123, 457, 3), "910e044b95eae9e6eafb8e317791e2358c54263c29129ea74dc45c3ce7bd66a4"),
            ((640, 480, 4), "f60a60aeba26276eec1877f9f8e150bcde89ccf4fe1d4209f3d3ab7f57ad9e71"),
            ((45, 45, 5), "03dc473f703cd9172f39e563689eb24441b0c0d0224d4943db2b5b1403c8ad97"),
            ((900, 31, 6), "581be971659cd775dffafe2e24e7fa1080df26538d82e728fcc42d204f646a34"),
        ];
        for ((width, height, seed), digest) in cases {
            let plan = plan(width, height, &CONFIG);
            let padded = pad(&noise(width, height, seed), plan.width, plan.height, [127; 3]);
            assert_eq!(format!("{:x}", Sha256::digest(&padded.pixels)), digest, "{width}x{height}");
        }
    }

    #[test]
    fn a_span_lays_out_rows_then_their_newline() {
        let slots = span(2, 3);
        assert_eq!(slots.len(), span_tokens(2, 3));
        assert_eq!(slots[0], Slot::Start);
        assert_eq!(slots[1..5], [Slot::Image(0), Slot::Image(1), Slot::Image(2), Slot::NewLine]);
        assert_eq!(slots[5..9], [Slot::Image(3), Slot::Image(4), Slot::Image(5), Slot::NewLine]);
        assert_eq!(slots[9], Slot::End);
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(bf16(1.0), 0x3f80);
        assert_eq!(bf16(-1.0), 0xbf80);
        assert_eq!(bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(bf16(f32::from_bits(0x3f80_8001)), 0x3f81);
    }

    #[test]
    fn resizing_to_the_same_size_copies() {
        let image = Rgb { width: 3, height: 2, pixels: (0..18).collect() };
        assert_eq!(resize(&image, 3, 2), image);
    }

    #[test]
    fn a_flat_image_stays_flat() {
        let image = Rgb { width: 37, height: 23, pixels: vec![200; 37 * 23 * 3] };
        for (width, height) in [(70, 50), (11, 9), (37, 61)] {
            let resized = resize(&image, width, height);
            assert!(resized.pixels.iter().all(|&value| value == 200), "{width}x{height}");
        }
    }
}
