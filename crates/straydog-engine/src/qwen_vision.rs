//! Images for Qwen3.8-Flash-Next (`qwen4_exp`, its vision tower that of `qwen3_5_moe`).
//!
//! An image is resized as the reference's `Qwen2VLImageProcessorFast` does (`smart_resize`, bicubic), cut into 16x16
//! patches in merge-block order (2x2 blocks in reading order, each block's four patches in reading order), and read by
//! 27 blocks of full attention with 2D RoPE on the CPU, the BF16 weights read where they are mapped and each product in
//! F32. Each 2x2 block of patches then comes out as one row of the text model's width. The prompt holds
//! `<|vision_start|>`, those rows, `<|vision_end|>`; [`rope_of`] says where each position turns under the text
//! model's interleaved mRoPE.
//!
//! The file (`mmproj-*.gguf`, the projector the GGUF repository ships) is checked against its pin at the start, read
//! through the page cache: nothing of it is held until an image comes, and its mapping goes when the image is encoded.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use memmap2::Mmap;
use rayon::prelude::*;

use crate::bundle::Asset;
use crate::vision::Rgb;

/// `<|vision_start|>`, `<|image_pad|>` (what PLE's n-grams hash at an image's positions) and `<|vision_end|>`.
pub const VISION_START: u32 = 248053;
pub const IMAGE_PAD: u32 = 248056;
pub const VISION_END: u32 = 248054;

const PATCH: usize = 16;
const MERGE: usize = 2;
/// A merged token's side in pixels: the size a resized image is a multiple of.
const FACTOR: usize = PATCH * MERGE;
/// The reference's `size.shortest_edge` in its `preprocessor_config.json`: the fewest pixels, in fact (64 tokens).
const MIN_PIXELS: usize = 65536;
/// The most pixels, 4096 tokens: llama.cpp's limit for this projector (`set_limit_image_tokens(8, 4096)`), under the
/// reference's `longest_edge` of 16777216 pixels, whose 65536 patches would take this encoder many minutes.
const MAX_PIXELS: usize = 4096 * FACTOR * FACTOR;
/// The reference refuses an aspect ratio beyond this.
const MAX_ASPECT: usize = 200;

const HIDDEN: usize = 1152;
const HEADS: usize = 16;
const HEAD: usize = HIDDEN / HEADS;
const QKV: usize = 3 * HIDDEN;
const FFN: usize = 4304;
const MERGED: usize = HIDDEN * MERGE * MERGE;
/// The text model's width: a merged token's row.
pub const ROW: usize = 2560;
const DEPTH: usize = 27;
const PATCH_VALUES: usize = 3 * PATCH * PATCH;
/// The position table's side (2304 rows).
const GRID: usize = 48;
/// RoPE's rotated values a head: all of them. Half turn by the row, half by the column.
const ROPE: usize = HEAD / 2;
const ROPE_THETA: f64 = 10000.0;
const LAYER_NORM_EPS: f32 = 1e-6;

// ---- preprocessing ----

/// The reference's `smart_resize` for this model (height, width), multiples of 32; ties round to even as Python's do.
pub fn smart_resize(height: usize, width: usize) -> Result<(usize, usize), String> {
    if height == 0 || width == 0 {
        return Err("an empty image".into());
    }
    if height.max(width) > MAX_ASPECT * height.min(width) {
        return Err(format!("an image of {width}x{height} pixels: its aspect ratio is over {MAX_ASPECT}"));
    }
    let (h, w, factor) = (height as f64, width as f64, FACTOR as f64);
    let (rounded_height, rounded_width) = ((h / factor).round_ties_even() as usize * FACTOR, (w / factor).round_ties_even() as usize * FACTOR);
    if rounded_height * rounded_width > MAX_PIXELS {
        let beta = (h * w / MAX_PIXELS as f64).sqrt();
        return Ok((FACTOR.max((h / beta / factor).floor() as usize * FACTOR), FACTOR.max((w / beta / factor).floor() as usize * FACTOR)));
    }
    if rounded_height * rounded_width < MIN_PIXELS {
        let beta = (MIN_PIXELS as f64 / (h * w)).sqrt();
        return Ok(((h * beta / factor).ceil() as usize * FACTOR, (w * beta / factor).ceil() as usize * FACTOR));
    }
    Ok((rounded_height, rounded_width))
}

/// An image ready for the encoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prepared {
    /// Its patch grid.
    pub rows: usize,
    pub columns: usize,
    rgb: Rgb,
}

impl Prepared {
    /// Its merged tokens' grid (height, width).
    pub fn grid(&self) -> (usize, usize) {
        (self.rows / MERGE, self.columns / MERGE)
    }

    pub fn tokens(&self) -> usize {
        self.rows * self.columns / (MERGE * MERGE)
    }
}

/// Resize an image to the encoder's size (bicubic, as Pillow; torch's on 8-bit values follows it within rounding).
pub fn prepare(image: &Rgb) -> Result<Prepared, String> {
    let (height, width) = smart_resize(image.height, image.width)?;
    Ok(Prepared { rows: height / PATCH, columns: width / PATCH, rgb: crate::vision::resize(image, width, height) })
}

/// The patches' (row, column) in the encoder's order: 2x2 blocks in reading order, each block's patches in reading order.
fn merge_order(rows: usize, columns: usize) -> Vec<(u32, u32)> {
    let mut order = Vec::with_capacity(rows * columns);
    for block_row in 0..rows / MERGE {
        for block_column in 0..columns / MERGE {
            for row in 0..MERGE {
                for column in 0..MERGE {
                    order.push(((block_row * MERGE + row) as u32, (block_column * MERGE + column) as u32));
                }
            }
        }
    }
    order
}

/// A position's two rows of the position table and their weights, along one axis of `size` patches (bilinear, corners aligned).
fn axis_taps(index: usize, size: usize) -> ([usize; 2], [f32; 2]) {
    let source = index as f64 * (GRID - 1) as f64 / (size.max(2) - 1) as f64;
    let floor = source.floor();
    let fraction = (source - floor) as f32;
    let low = floor as usize;
    ([low.min(GRID - 1), (low + 1).min(GRID - 1)], [1.0 - fraction, fraction])
}

// ---- the text model's positions ----

/// An image's span in a prompt.
#[derive(Debug, Clone)]
pub struct ImageSpan {
    /// Its first position (past its `<|vision_start|>`).
    pub start: usize,
    /// Its merged tokens' grid (height, width).
    pub grid: (usize, usize),
    /// Its rows, [`ROW`] values each, in reading order.
    pub rows: Arc<Vec<f32>>,
}

impl ImageSpan {
    fn tokens(&self) -> usize {
        self.grid.0 * self.grid.1
    }

    fn end(&self) -> usize {
        self.start + self.tokens()
    }

    /// What it takes from the rope positions of the text after it: the reference's `current_pos` moves on by
    /// the grid's longer side, not by its tokens.
    fn delta(&self) -> usize {
        self.tokens() - self.grid.0.max(self.grid.1)
    }
}

/// A position's (t, h, w) under mRoPE (`get_rope_index`). Text has one position on all three, its index less what the
/// images before it took; an image's tokens have the time of its first and their row and column on from there.
pub fn rope_of(spans: &[ImageSpan], position: usize) -> [u32; 3] {
    let mut delta = 0;
    for span in spans {
        if position < span.start {
            break;
        }
        if position < span.end() {
            let (offset, width, base) = (position - span.start, span.grid.1, span.start - delta);
            return [base as u32, (base + offset / width) as u32, (base + offset % width) as u32];
        }
        delta += span.delta();
    }
    [(position - delta) as u32; 3]
}

/// What a text position at or past `position` is ahead of its rope position, the images ended by then counted.
pub fn rope_delta(spans: &[ImageSpan], position: usize) -> u32 {
    spans.iter().take_while(|span| span.end() <= position).map(ImageSpan::delta).sum::<usize>() as u32
}

// ---- the file ----

/// A tensor's place in the file.
#[derive(Debug, Clone, Copy)]
struct Place {
    offset: usize,
    bytes: usize,
}

#[derive(Debug, Clone, Copy)]
struct BlockPlaces {
    qkv: (Place, Place),
    out: (Place, Place),
    up: (Place, Place),
    down: (Place, Place),
    norm1: (Place, Place),
    norm2: (Place, Place),
}

#[derive(Debug, Clone)]
struct Layout {
    patch: [Place; 2],
    patch_bias: Place,
    positions: Place,
    blocks: Vec<BlockPlaces>,
    post_norm: (Place, Place),
    merge_up: (Place, Place),
    merge_down: (Place, Place),
}

const F32_KIND: u32 = 0;
const BF16_KIND: u32 = 30;

fn layout(header: &straydog_gguf::ModelHeader) -> Result<Layout, String> {
    let text = |key: &str| header.metadata.get(key).and_then(|value| value.text());
    let number = |key: &str| header.metadata.get(key).and_then(|value| value.integer());
    if header.architecture != "clip" || text("clip.projector_type") != Some("qwen3vl_merger") {
        return Err("it is not a Qwen vision projector".into());
    }
    for (key, value) in [("clip.vision.patch_size", PATCH), ("clip.vision.embedding_length", HIDDEN), ("clip.vision.block_count", DEPTH),
                         ("clip.vision.attention.head_count", HEADS), ("clip.vision.feed_forward_length", FFN),
                         ("clip.vision.projection_dim", ROW), ("clip.vision.spatial_merge_size", MERGE)] {
        if number(key) != Some(value as i64) {
            return Err(format!("its {key} is not {value}"));
        }
    }
    let find = |name: &str, dims: &[u64], kind: u32| -> Result<Place, String> {
        let tensor = header.tensors.iter().find(|tensor| tensor.name == name).ok_or_else(|| format!("it has no tensor {name}"))?;
        if tensor.dimensions != dims || tensor.quant_type != kind {
            return Err(format!("its tensor {name} is {:?} of kind {}, not {dims:?} of kind {kind}", tensor.dimensions, tensor.quant_type));
        }
        // (mapped and read as what they hold: F32 and BF16 values on their own boundaries)
        if !tensor.offset.is_multiple_of(4) {
            return Err(format!("its tensor {name} is not aligned"));
        }
        Ok(Place { offset: tensor.offset as usize, bytes: tensor.bytes as usize })
    };
    let (hidden, qkv, ffn, merged, row) = (HIDDEN as u64, QKV as u64, FFN as u64, MERGED as u64, ROW as u64);
    let matrix = |name: &str, cols: u64, rows: u64| -> Result<(Place, Place), String> {
        Ok((find(&format!("{name}.weight"), &[cols, rows], BF16_KIND)?, find(&format!("{name}.bias"), &[rows], F32_KIND)?))
    };
    let norm = |name: &str| -> Result<(Place, Place), String> {
        Ok((find(&format!("{name}.weight"), &[hidden], F32_KIND)?, find(&format!("{name}.bias"), &[hidden], F32_KIND)?))
    };
    let mut blocks = Vec::with_capacity(DEPTH);
    for block in 0..DEPTH {
        let name = |part: &str| format!("v.blk.{block}.{part}");
        blocks.push(BlockPlaces {
            qkv: matrix(&name("attn_qkv"), hidden, qkv)?,
            out: matrix(&name("attn_out"), hidden, hidden)?,
            up: matrix(&name("ffn_up"), hidden, ffn)?,
            down: matrix(&name("ffn_down"), ffn, hidden)?,
            norm1: norm(&name("ln1"))?,
            norm2: norm(&name("ln2"))?,
        });
    }
    let patch_dims = [PATCH as u64, PATCH as u64, 3, hidden];
    Ok(Layout {
        patch: [find("v.patch_embd.weight", &patch_dims, F32_KIND)?, find("v.patch_embd.weight.1", &patch_dims, F32_KIND)?],
        patch_bias: find("v.patch_embd.bias", &[hidden], F32_KIND)?,
        positions: find("v.position_embd.weight", &[hidden, (GRID * GRID) as u64], F32_KIND)?,
        blocks,
        post_norm: norm("v.post_ln")?,
        merge_up: matrix("mm.0", merged, merged)?,
        merge_down: matrix("mm.2", merged, row)?,
    })
}

type Signature = (u64, Option<SystemTime>);

fn signature(file: &File) -> std::io::Result<Signature> {
    let metadata = file.metadata()?;
    Ok((metadata.len(), metadata.modified().ok()))
}

/// The vision tower on the CPU. Another backend takes the same [`Prepared`] and gives the same rows.
pub struct Encoder {
    path: PathBuf,
    file: File,
    signature: Signature,
    layout: Layout,
    /// The threads an image is encoded on.
    pub threads: usize,
}

impl Encoder {
    /// Open the projector at `path`, authenticated against its pin: every byte is hashed through the page cache, so
    /// nothing stays in this process. Err says what is wrong with the file.
    pub fn open(path: &Path, pin: &Asset) -> Result<Encoder, String> {
        let shown = path.display();
        let file = File::open(path).map_err(|exc| format!("the vision encoder {shown}: {exc}"))?;
        let before = signature(&file).map_err(|exc| format!("the vision encoder {shown}: {exc}"))?;
        if before.0 != pin.bytes {
            return Err(format!("the vision encoder {shown} has {} bytes, not the {} of the pinned {}", before.0, pin.bytes, pin.name));
        }
        let header = straydog_gguf::inspect(&mut std::io::BufReader::new(&file)).map_err(|exc| format!("the vision encoder {shown}: {exc}"))?;
        let layout = layout(&header).map_err(|message| format!("the vision encoder {shown}: {message}"))?;
        let digest = crate::digest::file_sha256(&file, &|_| {}).map_err(|exc| format!("the vision encoder {shown}: {exc}"))?;
        if digest != pin.sha256 {
            return Err(format!("the vision encoder {shown} is not the pinned {}: its SHA-256 is {digest}", pin.name));
        }
        let threads = std::thread::available_parallelism().map_or(8, |count| count.get());
        Ok(Encoder { path: path.to_path_buf(), file, signature: before, layout, threads })
    }

    /// An image's rows, [`ROW`] values each, in reading order; None when `cancelled` said so between blocks.
    pub fn encode(&self, image: &Prepared, cancelled: &(dyn Fn() -> bool + Sync)) -> Result<Option<Vec<f32>>, String> {
        let shown = self.path.display();
        if signature(&self.file).map_err(|exc| format!("{shown}: {exc}"))? != self.signature {
            return Err(format!("the vision encoder {shown} changed on disk"));
        }
        // (buffers a request can hardly do without: a refusal rather than the process's end)
        let needed = image.rows * image.columns * WORK_FLOATS * 4;
        if let Some((available, commit)) = crate::host::commit_left() {
            if (needed as u64).saturating_add(MEMORY_MARGIN) > available.min(commit) {
                return Err(format!("an image of {}x{} needs {:.1} GiB to encode, more than the machine has free",
                                   image.columns * PATCH, image.rows * PATCH, needed as f64 / (1u64 << 30) as f64));
            }
        }
        // SAFETY: mapped read-only; the file is pinned, checked unchanged above, and the mapping ends with this call.
        let map = unsafe { Mmap::map(&self.file) }.map_err(|exc| format!("{shown}: {exc}"))?;
        let pool = rayon::ThreadPoolBuilder::new().num_threads(self.threads).build().map_err(|exc| exc.to_string())?;
        let tower = Tower::new(&map, &self.layout)?;
        let rows = pool.install(|| tower.encode(image, cancelled))?;
        match &rows {
            Some(rows) if !rows.iter().all(|value| value.is_finite()) => Err("the vision encoder's rows are not finite".into()),
            _ => Ok(rows),
        }
    }
}

/// The floats a patch takes while encoding: the hidden rows and their norms, QKV, keys by column, values by head, attention
/// out, the branch, the feed-forward's middle.
const WORK_FLOATS: usize = 6 * HIDDEN + QKV + FFN;
/// What an image's encoding leaves the machine.
const MEMORY_MARGIN: u64 = 1 << 30;

fn zeros(count: usize) -> Result<Vec<f32>, String> {
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|_| format!("no memory for {:.1} GiB of the vision encoder's rows", (count * 4) as f64 / (1u64 << 30) as f64))?;
    values.resize(count, 0.0);
    Ok(values)
}

// ---- the arithmetic ----

/// The widest vectors this CPU runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Level {
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    Plain,
}

fn level() -> Level {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512dq") && is_x86_feature_detected!("avx512vl") {
            return Level::Avx512;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Level::Avx2;
        }
    }
    Level::Plain
}

/// `$name(level, args)`: `$body` compiled for each level's vectors.
macro_rules! by_level {
    ($name:ident, $wide:ident, $narrow:ident, $body:ident, ($($arg:ident: $kind:ty),* $(,)?)) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
        unsafe fn $wide($($arg: $kind),*) {
            $body($($arg),*)
        }
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn $narrow($($arg: $kind),*) {
            $body($($arg),*)
        }
        #[allow(clippy::too_many_arguments)]
        fn $name(level: Level, $($arg: $kind),*) {
            match level {
                // SAFETY: `level` found these features on this CPU.
                #[cfg(target_arch = "x86_64")]
                Level::Avx512 => unsafe { $wide($($arg),*) },
                // SAFETY: as above.
                #[cfg(target_arch = "x86_64")]
                Level::Avx2 => unsafe { $narrow($($arg),*) },
                Level::Plain => $body($($arg),*),
            }
        }
    };
}

/// BF16 to F32: its bits are an F32's upper half.
#[inline(always)]
fn widen(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

/// Rows of the dense weights a thread goes through against a tile of tokens: with [`TILE_TOKENS`], what its second-level cache holds.
const TILE_ROWS: usize = 16;
const TILE_TOKENS: usize = 64;

/// The four rows' products with `x`.
#[inline(always)]
fn dot4_plain(rows: [&[u16]; 4], x: &[f32]) -> [f32; 4] {
    const LANES: usize = 16;
    let mut sums = [[0f32; LANES]; 4];
    let whole = x.len() / LANES * LANES;
    for at in (0..whole).step_by(LANES) {
        let values: &[f32; LANES] = x[at..at + LANES].try_into().unwrap();
        for (row, sum) in rows.iter().zip(&mut sums) {
            let weights: &[u16; LANES] = row[at..at + LANES].try_into().unwrap();
            for lane in 0..LANES {
                sum[lane] += widen(weights[lane]) * values[lane];
            }
        }
    }
    let mut totals = [0f32; 4];
    for (index, total) in totals.iter_mut().enumerate() {
        *total = sums[index].iter().sum::<f32>() + rows[index][whole..].iter().zip(&x[whole..]).map(|(weight, value)| widen(*weight) * value).sum::<f32>();
    }
    totals
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn dot4_wide(rows: [&[u16]; 4], x: &[f32]) -> [f32; 4] {
    use std::arch::x86_64::*;
    let whole = x.len() / 16 * 16;
    let mut sums = [_mm512_setzero_ps(); 4];
    for at in (0..whole).step_by(16) {
        // SAFETY: the rows and `x` are of one length (the callers' slices), `at + 16` within it.
        unsafe {
            let values = _mm512_loadu_ps(x.as_ptr().add(at));
            for (row, sum) in rows.iter().zip(&mut sums) {
                let halves = _mm256_loadu_si256(row.as_ptr().add(at).cast());
                let weights = _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(halves)));
                *sum = _mm512_fmadd_ps(weights, values, *sum);
            }
        }
    }
    let mut totals = [0f32; 4];
    for (index, total) in totals.iter_mut().enumerate() {
        *total = _mm512_reduce_add_ps(sums[index]) + rows[index][whole..].iter().zip(&x[whole..]).map(|(weight, value)| widen(*weight) * value).sum::<f32>();
    }
    totals
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot4_narrow(rows: [&[u16]; 4], x: &[f32]) -> [f32; 4] {
    use std::arch::x86_64::*;
    let whole = x.len() / 8 * 8;
    let mut sums = [_mm256_setzero_ps(); 4];
    for at in (0..whole).step_by(8) {
        // SAFETY: as in `dot4_wide`.
        unsafe {
            let values = _mm256_loadu_ps(x.as_ptr().add(at));
            for (row, sum) in rows.iter().zip(&mut sums) {
                let halves = _mm_loadu_si128(row.as_ptr().add(at).cast());
                let weights = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(halves)));
                *sum = _mm256_fmadd_ps(weights, values, *sum);
            }
        }
    }
    let mut totals = [0f32; 4];
    for (index, total) in totals.iter_mut().enumerate() {
        let mut lanes = [0f32; 8];
        // SAFETY: eight values into eight.
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), sums[index]) };
        *total = lanes.iter().sum::<f32>() + rows[index][whole..].iter().zip(&x[whole..]).map(|(weight, value)| widen(*weight) * value).sum::<f32>();
    }
    totals
}

/// `$dot` over a tile: `y` (tokens x rows) from `weights` (rows x cols, BF16) and `x` (tokens x cols), plus the bias.
macro_rules! tile_body {
    ($dot:expr, $weights:ident, $rows:ident, $cols:ident, $bias:ident, $x:ident, $y:ident) => {{
        let row_of = |row: usize| &$weights[row * $cols..(row + 1) * $cols];
        let tokens = $x.len() / $cols;
        for first in (0..$rows).step_by(TILE_ROWS) {
            let end = (first + TILE_ROWS).min($rows);
            for token in 0..tokens {
                let values = &$x[token * $cols..(token + 1) * $cols];
                let target = &mut $y[token * $rows + first..token * $rows + end];
                let mut row = first;
                while row < end {
                    // (a last group short of four repeats its last row, its other products unused)
                    let group = [row_of(row), row_of((row + 1).min(end - 1)), row_of((row + 2).min(end - 1)), row_of((row + 3).min(end - 1))];
                    let totals = $dot(group, values);
                    for (index, total) in totals.iter().enumerate().take(end - row) {
                        target[row - first + index] = total + $bias[row + index];
                    }
                    row += 4;
                }
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn tile_wide(weights: &[u16], rows: usize, cols: usize, bias: &[f32], x: &[f32], y: &mut [f32]) {
    // SAFETY: this function's own features.
    tile_body!(|group, values| unsafe { dot4_wide(group, values) }, weights, rows, cols, bias, x, y)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tile_narrow(weights: &[u16], rows: usize, cols: usize, bias: &[f32], x: &[f32], y: &mut [f32]) {
    // SAFETY: this function's own features.
    tile_body!(|group, values| unsafe { dot4_narrow(group, values) }, weights, rows, cols, bias, x, y)
}

fn tile_plain(weights: &[u16], rows: usize, cols: usize, bias: &[f32], x: &[f32], y: &mut [f32]) {
    tile_body!(dot4_plain, weights, rows, cols, bias, x, y)
}

/// `y` (tokens x rows) = `x` (tokens x cols) through the BF16 `weights` (rows x cols), plus `bias`, every token on a core of its tile.
fn dense(level: Level, weights: &[u16], rows: usize, cols: usize, bias: &[f32], x: &[f32], y: &mut [f32]) {
    assert!(weights.len() == rows * cols && bias.len() == rows && x.len() / cols == y.len() / rows);
    y.par_chunks_mut(rows * TILE_TOKENS).zip(x.par_chunks(cols * TILE_TOKENS)).for_each(|(y, x)| match level {
        // SAFETY: `level` found these features on this CPU.
        #[cfg(target_arch = "x86_64")]
        Level::Avx512 => unsafe { tile_wide(weights, rows, cols, bias, x, y) },
        // SAFETY: as above.
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 => unsafe { tile_narrow(weights, rows, cols, bias, x, y) },
        Level::Plain => tile_plain(weights, rows, cols, bias, x, y),
    });
}

/// e^x for x <= 0, within two units in the last place (Cephes' expf); underflow goes to about 1e-38.
#[inline(always)]
fn exp_negative(x: f32) -> f32 {
    // (a NaN goes to the low end: what follows must see a number)
    let x = if x >= -87.3 { x } else { -87.3 };
    let scaled = x * std::f32::consts::LOG2_E - 0.5;
    // SAFETY: a number within i32: x is in [-87.3, 0], the callers' arguments being differences from a greatest value or negated magnitudes.
    let whole: i32 = unsafe { scaled.to_int_unchecked() };
    let n = whole as f32;
    let r = x - n * 0.693_359_375 - n * -2.121_944_4e-4;
    let tail = ((((1.987_569_1e-4 * r + 1.398_199_9e-3) * r + 8.333_452e-3) * r + 4.166_579_6e-2) * r + 1.666_666_5e-1) * r + 5.000_000_3e-1;
    let value = tail * r * r + r + 1.0;
    f32::from_bits(((whole + 127) as u32) << 23) * value
}

/// `gelu_pytorch_tanh`: `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`.
#[inline(always)]
fn gelu_tanh(x: f32) -> f32 {
    const SCALE: f32 = 0.797_884_6; // sqrt(2 / pi)
    let inner = SCALE * (x + 0.044715 * x * x * x);
    let magnitude = inner.abs();
    let fall = exp_negative(-2.0 * magnitude);
    let tanh = (1.0 - fall) / (1.0 + fall);
    let tanh = f32::from_bits(tanh.to_bits() | (inner.to_bits() & 0x8000_0000));
    0.5 * x * (1.0 + tanh)
}

/// erf within about 1e-7 (Numerical Recipes' `erfcc`).
fn erf(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let tail = t * (-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806
        + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))).exp();
    if x >= 0.0 { 1.0 - tail } else { tail - 1.0 }
}

/// `nn.GELU()`: `0.5 * x * (1 + erf(x / sqrt(2)))`.
fn gelu(x: f32) -> f32 {
    let x = f64::from(x);
    (0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2))) as f32
}

#[inline(always)]
fn gelu_tanh_body(values: &mut [f32]) {
    for value in values {
        *value = gelu_tanh(*value);
    }
}
by_level!(gelu_tanh_all, gelu_tanh_wide, gelu_tanh_narrow, gelu_tanh_body, (values: &mut [f32]));

/// A row's `nn.LayerNorm`.
#[inline(always)]
fn layer_norm_body(x: &[f32], weight: &[f32], bias: &[f32], y: &mut [f32]) {
    const LANES: usize = 16;
    #[inline(always)]
    fn sum(values: &[f32], each: impl Fn(f32) -> f32) -> f32 {
        let mut lanes = [0f32; LANES];
        let chunks = values.chunks_exact(LANES);
        let rest: f32 = chunks.remainder().iter().map(|value| each(*value)).sum();
        for chunk in chunks {
            for lane in 0..LANES {
                lanes[lane] += each(chunk[lane]);
            }
        }
        lanes.iter().sum::<f32>() + rest
    }
    let count = x.len() as f32;
    let mean = sum(x, |value| value) / count;
    let variance = sum(x, |value| (value - mean) * (value - mean)) / count;
    let scale = 1.0 / (variance + LAYER_NORM_EPS).sqrt();
    for (((target, value), weight), bias) in y.iter_mut().zip(x).zip(weight).zip(bias) {
        *target = (value - mean) * scale * weight + bias;
    }
}
by_level!(layer_norm, layer_norm_wide, layer_norm_narrow, layer_norm_body, (x: &[f32], weight: &[f32], bias: &[f32], y: &mut [f32]));

/// The greatest of `values` and `peak`.
#[inline(always)]
fn peak_of(values: &[f32], peak: f32) -> f32 {
    const LANES: usize = 16;
    let mut peaks = [peak; LANES];
    let chunks = values.chunks_exact(LANES);
    let mut peak = chunks.remainder().iter().fold(peak, |peak, value| if *value > peak { *value } else { peak });
    for chunk in chunks {
        for lane in 0..LANES {
            peaks[lane] = if chunk[lane] > peaks[lane] { chunk[lane] } else { peaks[lane] };
        }
    }
    for value in peaks {
        peak = peak.max(value);
    }
    peak
}

/// A piece of a query's softmax, as torch writes it (less the greatest first): `exp((scores - peak) * scale)`, and their sum.
#[inline(always)]
fn exponentials(scores: &mut [f32], peak: f32, scale: f32) -> f32 {
    const LANES: usize = 16;
    for value in scores.iter_mut() {
        *value = exp_negative((*value - peak) * scale);
    }
    let mut totals = [0f32; LANES];
    let chunks = scores.chunks_exact(LANES);
    let rest: f32 = chunks.remainder().iter().sum();
    for chunk in chunks {
        for lane in 0..LANES {
            totals[lane] += chunk[lane];
        }
    }
    totals.iter().sum::<f32>() + rest
}

/// A patch's first row: its values through both temporal filters (a still image is two equal frames), plus the bias.
#[inline(always)]
fn filter_patch_body(first: &[f32], second: &[f32], bias: &[f32], patch: &[f32], target: &mut [f32]) {
    const LANES: usize = 16;
    for (out, target) in target.iter_mut().enumerate() {
        let (low, high) = (&first[out * PATCH_VALUES..(out + 1) * PATCH_VALUES], &second[out * PATCH_VALUES..(out + 1) * PATCH_VALUES]);
        let mut sums = [0f32; LANES];
        for ((low, high), value) in low.chunks_exact(LANES).zip(high.chunks_exact(LANES)).zip(patch.chunks_exact(LANES)) {
            for lane in 0..LANES {
                sums[lane] += (low[lane] + high[lane]) * value[lane];
            }
        }
        *target = sums.iter().sum::<f32>() + bias[out];
    }
}
by_level!(filter_patch, filter_patch_wide, filter_patch_narrow, filter_patch_body,
          (first: &[f32], second: &[f32], bias: &[f32], patch: &[f32], target: &mut [f32]));

/// Keys and values a thread goes through at once against [`QUERY_TILE`] queries (a 1440x900 screen's attention took 8.5 s at 256 keys,
/// 6.4 at 128, 5.7 at 64, 7.3 at 32).
const KEY_TILE: usize = 64;
const QUERY_TILE: usize = 32;

/// The floats of a head's keys laid out by tile.
fn keys_floats(count: usize) -> usize {
    count.div_ceil(KEY_TILE) * HEAD * KEY_TILE
}

/// `target += factors[0] * columns[0] + ...`. Four columns to a store: what a core's two multiplies a cycle want.
fn score4_plain(target: &mut [f32], columns: [&[f32]; 4], factors: [f32; 4]) {
    for (index, score) in target.iter_mut().enumerate() {
        *score += factors[0] * columns[0][index] + factors[1] * columns[1][index] + factors[2] * columns[2][index] + factors[3] * columns[3][index];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn score4_wide(target: &mut [f32], columns: [&[f32]; 4], factors: [f32; 4]) {
    use std::arch::x86_64::*;
    let whole = target.len() / 16 * 16;
    let scale = factors.map(|factor| _mm512_set1_ps(factor));
    for at in (0..whole).step_by(16) {
        // SAFETY: the columns are as long as `target` (the caller's slices), `at + 16` within it.
        unsafe {
            let mut sum = _mm512_loadu_ps(target.as_ptr().add(at));
            for (column, factor) in columns.iter().zip(&scale) {
                sum = _mm512_fmadd_ps(*factor, _mm512_loadu_ps(column.as_ptr().add(at)), sum);
            }
            _mm512_storeu_ps(target.as_mut_ptr().add(at), sum);
        }
    }
    for at in whole..target.len() {
        target[at] += factors[0] * columns[0][at] + factors[1] * columns[1][at] + factors[2] * columns[2][at] + factors[3] * columns[3][at];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn score4_narrow(target: &mut [f32], columns: [&[f32]; 4], factors: [f32; 4]) {
    use std::arch::x86_64::*;
    let whole = target.len() / 8 * 8;
    let scale = factors.map(|factor| _mm256_set1_ps(factor));
    for at in (0..whole).step_by(8) {
        // SAFETY: as in `score4_wide`.
        unsafe {
            let mut sum = _mm256_loadu_ps(target.as_ptr().add(at));
            for (column, factor) in columns.iter().zip(&scale) {
                sum = _mm256_fmadd_ps(*factor, _mm256_loadu_ps(column.as_ptr().add(at)), sum);
            }
            _mm256_storeu_ps(target.as_mut_ptr().add(at), sum);
        }
    }
    for at in whole..target.len() {
        target[at] += factors[0] * columns[0][at] + factors[1] * columns[1][at] + factors[2] * columns[2][at] + factors[3] * columns[3][at];
    }
}

/// `target += weights[0] * rows[0..HEAD] + ...` over four rows of values.
fn gather4_plain(target: &mut [f32; HEAD], rows: &[f32; 4 * HEAD], weights: [f32; 4]) {
    for dimension in 0..HEAD {
        target[dimension] += weights[0] * rows[dimension] + weights[1] * rows[HEAD + dimension] + weights[2] * rows[2 * HEAD + dimension] + weights[3] * rows[3 * HEAD + dimension];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2,fma")]
#[inline]
unsafe fn gather4_wide(target: &mut [f32; HEAD], rows: &[f32; 4 * HEAD], weights: [f32; 4]) {
    use std::arch::x86_64::*;
    // (a head's 72 values: four registers of 16 and one of 8)
    let wide = weights.map(|weight| _mm512_set1_ps(weight));
    let narrow = weights.map(|weight| _mm256_set1_ps(weight));
    // SAFETY: within the arrays: 64 + 8 is a head, each row a head from the last.
    unsafe {
        for at in (0..64).step_by(16) {
            let mut sum = _mm512_loadu_ps(target.as_ptr().add(at));
            for (row, weight) in wide.iter().enumerate() {
                sum = _mm512_fmadd_ps(*weight, _mm512_loadu_ps(rows.as_ptr().add(row * HEAD + at)), sum);
            }
            _mm512_storeu_ps(target.as_mut_ptr().add(at), sum);
        }
        let mut sum = _mm256_loadu_ps(target.as_ptr().add(64));
        for (row, weight) in narrow.iter().enumerate() {
            sum = _mm256_fmadd_ps(*weight, _mm256_loadu_ps(rows.as_ptr().add(row * HEAD + 64)), sum);
        }
        _mm256_storeu_ps(target.as_mut_ptr().add(64), sum);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn gather4_narrow(target: &mut [f32; HEAD], rows: &[f32; 4 * HEAD], weights: [f32; 4]) {
    use std::arch::x86_64::*;
    let scale = weights.map(|weight| _mm256_set1_ps(weight));
    for at in (0..HEAD).step_by(8) {
        // SAFETY: within the arrays: a head is nine registers of 8.
        unsafe {
            let mut sum = _mm256_loadu_ps(target.as_ptr().add(at));
            for (row, weight) in scale.iter().enumerate() {
                sum = _mm256_fmadd_ps(*weight, _mm256_loadu_ps(rows.as_ptr().add(row * HEAD + at)), sum);
            }
            _mm256_storeu_ps(target.as_mut_ptr().add(at), sum);
        }
    }
}

/// A tile of queries of one head: `out[q]` (a head's width each) from their query rows (`$queries`, `$query_stride` apart), the
/// head's keys by tile, each by column (`keys[t * HEAD * KEY_TILE + d * KEY_TILE + j]`, [`keys_floats`]), and its values by row
/// (`values[j * HEAD + d]`), through `$score` and `$gather`. Everything of a tile lies together: spread by whole columns, the
/// first-level translation buffer's entries went on 72 pages a tile, and a smaller tile took longer than a larger one.
macro_rules! attend_body {
    ($score:expr, $gather:expr, $queries:ident, $query_stride:ident, $keys:ident, $values:ident, $count:ident, $scores:ident, $out:ident) => {{
        let queries_here = $queries.len().div_ceil($query_stride);
        let (key_tiles, score_tile) = ($count.div_ceil(KEY_TILE), QUERY_TILE * KEY_TILE);
        for tile in 0..key_tiles {
            let length = KEY_TILE.min($count - tile * KEY_TILE);
            let block = &$keys[tile * HEAD * KEY_TILE..(tile + 1) * HEAD * KEY_TILE];
            for query in 0..queries_here {
                let target = &mut $scores[tile * score_tile + query * KEY_TILE..][..length];
                target.fill(0.0);
                let vector = &$queries[query * $query_stride..query * $query_stride + HEAD];
                for (group, factors) in vector.chunks_exact(4).enumerate() {
                    let column = |at: usize| &block[(group * 4 + at) * KEY_TILE..][..length];
                    $score(&mut *target, [column(0), column(1), column(2), column(3)], [factors[0], factors[1], factors[2], factors[3]]);
                }
            }
        }
        // A query's softmax over its pieces, tile by tile.
        let scale = 1.0 / (HEAD as f32).sqrt();
        let mut peaks = [f32::MIN; QUERY_TILE];
        let mut totals = [0f32; QUERY_TILE];
        for tile in 0..key_tiles {
            let length = KEY_TILE.min($count - tile * KEY_TILE);
            for query in 0..queries_here {
                peaks[query] = peak_of(&$scores[tile * score_tile + query * KEY_TILE..][..length], peaks[query]);
            }
        }
        for tile in 0..key_tiles {
            let length = KEY_TILE.min($count - tile * KEY_TILE);
            for query in 0..queries_here {
                totals[query] += exponentials(&mut $scores[tile * score_tile + query * KEY_TILE..][..length], peaks[query], scale);
            }
        }
        $out.fill(0.0);
        for tile in 0..key_tiles {
            let (first, length) = (tile * KEY_TILE, KEY_TILE.min($count - tile * KEY_TILE));
            let rows = &$values[first * HEAD..(first + length) * HEAD];
            let whole = length / 4 * 4;
            for query in 0..queries_here {
                let probabilities = &$scores[tile * score_tile + query * KEY_TILE..][..length];
                let inverse = 1.0 / totals[query];
                let target: &mut [f32; HEAD] = (&mut $out[query * HEAD..(query + 1) * HEAD]).try_into().unwrap();
                for (group, four) in probabilities.chunks_exact(4).zip(rows.chunks_exact(4 * HEAD)) {
                    $gather(&mut *target, four.try_into().unwrap(), [group[0] * inverse, group[1] * inverse, group[2] * inverse, group[3] * inverse]);
                }
                for (probability, row) in probabilities[whole..].iter().zip(rows[whole * HEAD..].chunks_exact(HEAD)) {
                    for (target, value) in target.iter_mut().zip(row) {
                        *target += probability * inverse * value;
                    }
                }
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
unsafe fn attend_wide(queries: &[f32], query_stride: usize, keys: &[f32], values: &[f32], count: usize, scores: &mut [f32], out: &mut [f32]) {
    // SAFETY: this function's own features.
    attend_body!(|t: &mut [f32], c, f| unsafe { score4_wide(t, c, f) }, |t: &mut [f32; HEAD], r, w| unsafe { gather4_wide(t, r, w) },
                 queries, query_stride, keys, values, count, scores, out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn attend_narrow(queries: &[f32], query_stride: usize, keys: &[f32], values: &[f32], count: usize, scores: &mut [f32], out: &mut [f32]) {
    // SAFETY: this function's own features.
    attend_body!(|t: &mut [f32], c, f| unsafe { score4_narrow(t, c, f) }, |t: &mut [f32; HEAD], r, w| unsafe { gather4_narrow(t, r, w) },
                 queries, query_stride, keys, values, count, scores, out)
}

fn attend_plain(queries: &[f32], query_stride: usize, keys: &[f32], values: &[f32], count: usize, scores: &mut [f32], out: &mut [f32]) {
    attend_body!(score4_plain, gather4_plain, queries, query_stride, keys, values, count, scores, out)
}

#[allow(clippy::too_many_arguments)]
fn attend(level: Level, queries: &[f32], query_stride: usize, keys: &[f32], values: &[f32], count: usize, scores: &mut [f32], out: &mut [f32]) {
    match level {
        // SAFETY: `level` found these features on this CPU.
        #[cfg(target_arch = "x86_64")]
        Level::Avx512 => unsafe { attend_wide(queries, query_stride, keys, values, count, scores, out) },
        // SAFETY: as above.
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 => unsafe { attend_narrow(queries, query_stride, keys, values, count, scores, out) },
        Level::Plain => attend_plain(queries, query_stride, keys, values, count, scores, out),
    }
}

/// RoPE of a token's query (or key) heads: each head's half turned against its other half by the token's angles.
#[inline(always)]
fn rope_body(heads: &mut [f32], cosines: &[f32], sines: &[f32]) {
    let cosines: &[f32; ROPE] = cosines.try_into().unwrap();
    let sines: &[f32; ROPE] = sines.try_into().unwrap();
    for head in heads.chunks_exact_mut(HEAD) {
        let (first, second) = head.split_at_mut(ROPE);
        let first: &mut [f32; ROPE] = first.try_into().unwrap();
        let second: &mut [f32; ROPE] = second.try_into().unwrap();
        for index in 0..ROPE {
            let (a, b) = (first[index], second[index]);
            first[index] = a * cosines[index] - b * sines[index];
            second[index] = b * cosines[index] + a * sines[index];
        }
    }
}
by_level!(rope, rope_wide, rope_narrow, rope_body, (heads: &mut [f32], cosines: &[f32], sines: &[f32]));

// ---- the tower ----

/// The file's tensors, mapped.
struct Tower<'a> {
    map: &'a [u8],
    layout: &'a Layout,
    level: Level,
}

impl<'a> Tower<'a> {
    fn new(map: &'a [u8], layout: &'a Layout) -> Result<Tower<'a>, String> {
        // (each file offset is a multiple of four: the mapping's page boundary keeps that)
        if !(map.as_ptr() as usize).is_multiple_of(4) {
            return Err("the vision encoder's mapping is not aligned".into());
        }
        Ok(Tower { map, layout, level: level() })
    }

    fn floats(&self, place: Place) -> &'a [f32] {
        let bytes = &self.map[place.offset..place.offset + place.bytes];
        // SAFETY: any four bytes are an F32, and Layout's offsets and the mapping's start are multiples of four.
        let (head, values, tail) = unsafe { bytes.align_to::<f32>() };
        assert!(head.is_empty() && tail.is_empty());
        values
    }

    fn halves(&self, place: Place) -> &'a [u16] {
        let bytes = &self.map[place.offset..place.offset + place.bytes];
        // SAFETY: as above, for BF16 bits.
        let (head, values, tail) = unsafe { bytes.align_to::<u16>() };
        assert!(head.is_empty() && tail.is_empty());
        values
    }

    fn norm_all(&self, (weight, bias): (Place, Place), x: &[f32], y: &mut [f32]) {
        let (weight, bias, level) = (self.floats(weight), self.floats(bias), self.level);
        y.par_chunks_mut(HIDDEN).zip(x.par_chunks(HIDDEN)).for_each(|(y, x)| layer_norm(level, x, weight, bias, y));
    }

    fn matrix(&self, (weight, bias): (Place, Place), rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
        dense(self.level, self.halves(weight), rows, cols, self.floats(bias), x, y);
    }

    /// The patches' first rows: the two temporal filters of the still image's two equal frames, the bias, the position table bilinearly.
    fn embed(&self, image: &Prepared, order: &[(u32, u32)], hidden: &mut [f32]) {
        let (first, second) = (self.floats(self.layout.patch[0]), self.floats(self.layout.patch[1]));
        let bias = self.floats(self.layout.patch_bias);
        let table = self.floats(self.layout.positions);
        let (rows, columns, width, pixels, level) = (image.rows, image.columns, image.rgb.width, &image.rgb.pixels, self.level);
        hidden.par_chunks_mut(HIDDEN).zip(order.par_iter()).for_each_init(|| vec![0f32; PATCH_VALUES], |patch, (target, &(row, column))| {
            let (row, column) = (row as usize, column as usize);
            for channel in 0..3 {
                for y in 0..PATCH {
                    for x in 0..PATCH {
                        let pixel = pixels[((row * PATCH + y) * width + column * PATCH + x) * 3 + channel];
                        patch[(channel * PATCH + y) * PATCH + x] = (f32::from(pixel) - 127.5) / 127.5;
                    }
                }
            }
            filter_patch(level, first, second, bias, patch, target);
            let (row_taps, row_weights) = axis_taps(row, rows);
            let (column_taps, column_weights) = axis_taps(column, columns);
            for (row_tap, row_weight) in row_taps.iter().zip(row_weights) {
                for (column_tap, column_weight) in column_taps.iter().zip(column_weights) {
                    let tap = &table[(row_tap * GRID + column_tap) * HIDDEN..][..HIDDEN];
                    let weight = row_weight * column_weight;
                    if weight != 0.0 {
                        for (target, value) in target.iter_mut().zip(tap) {
                            *target += weight * value;
                        }
                    }
                }
            }
        });
    }

    fn encode(&self, image: &Prepared, cancelled: &(dyn Fn() -> bool + Sync)) -> Result<Option<Vec<f32>>, String> {
        let count = image.rows * image.columns;
        let order = merge_order(image.rows, image.columns);
        let (cosines, sines) = rope_angles(&order);
        let mut hidden = zeros(count * HIDDEN)?;
        self.embed(image, &order, &mut hidden);
        let mut norm = zeros(count * HIDDEN)?;
        let mut qkv = zeros(count * QKV)?;
        let mut keys = zeros(HEADS * keys_floats(count))?;
        let mut values = zeros(count * HIDDEN)?;
        let mut attended = zeros(count * HIDDEN)?;
        let mut middle = zeros(count * FFN)?;
        let mut branch = zeros(count * HIDDEN)?;
        let level = self.level;
        for block in &self.layout.blocks {
            if cancelled() {
                return Ok(None);
            }
            self.norm_all(block.norm1, &hidden, &mut norm);
            self.matrix(block.qkv, QKV, HIDDEN, &norm, &mut qkv);
            // The queries and keys turned, then each head's keys by column and values by row.
            qkv.par_chunks_mut(QKV).enumerate().for_each(|(token, row)| {
                let (cosine, sine) = (&cosines[token * ROPE..(token + 1) * ROPE], &sines[token * ROPE..(token + 1) * ROPE]);
                rope(level, &mut row[..2 * HIDDEN], cosine, sine);
            });
            let qkv_view = &qkv;
            keys.par_chunks_mut(keys_floats(count)).enumerate().for_each(|(head, target)| {
                for (tile, block) in target.chunks_exact_mut(HEAD * KEY_TILE).enumerate() {
                    for (dimension, column) in block.chunks_exact_mut(KEY_TILE).enumerate() {
                        let tokens = tile * KEY_TILE..((tile + 1) * KEY_TILE).min(count);
                        for (value, token) in column.iter_mut().zip(tokens) {
                            *value = qkv_view[token * QKV + HIDDEN + head * HEAD + dimension];
                        }
                    }
                }
            });
            values.par_chunks_mut(HEAD * count).enumerate().for_each(|(head, target)| {
                for (token, row) in target.chunks_exact_mut(HEAD).enumerate() {
                    row.copy_from_slice(&qkv_view[token * QKV + 2 * HIDDEN + head * HEAD..][..HEAD]);
                }
            });
            let (keys_view, values_view) = (&keys, &values);
            attended.par_chunks_mut(HIDDEN * QUERY_TILE).enumerate().for_each_init(
                || (vec![0f32; count.div_ceil(KEY_TILE) * QUERY_TILE * KEY_TILE], vec![0f32; QUERY_TILE * HEAD]),
                |(scores, out), (tile, target)| {
                    let tokens = target.len() / HIDDEN;
                    for head in 0..HEADS {
                        let queries = &qkv_view[tile * QUERY_TILE * QKV + head * HEAD..(tile * QUERY_TILE + tokens - 1) * QKV + (head + 1) * HEAD];
                        attend(level, queries, QKV, &keys_view[head * keys_floats(count)..(head + 1) * keys_floats(count)],
                               &values_view[head * HEAD * count..(head + 1) * HEAD * count], count, scores, &mut out[..tokens * HEAD]);
                        for token in 0..tokens {
                            target[token * HIDDEN + head * HEAD..][..HEAD].copy_from_slice(&out[token * HEAD..(token + 1) * HEAD]);
                        }
                    }
                },
            );
            self.matrix(block.out, HIDDEN, HIDDEN, &attended, &mut branch);
            add_all(level, &branch, &mut hidden);
            self.norm_all(block.norm2, &hidden, &mut norm);
            self.matrix(block.up, FFN, HIDDEN, &norm, &mut middle);
            middle.par_chunks_mut(FFN).for_each(|row| gelu_tanh_all(level, row));
            self.matrix(block.down, HIDDEN, FFN, &middle, &mut branch);
            add_all(level, &branch, &mut hidden);
        }
        drop((qkv, keys, values, attended, branch));
        // The merger: each block's four patches in one row, its first product through GELU and its second to the text model's width.
        self.norm_all(self.layout.post_norm, &hidden, &mut norm);
        let tokens = count / (MERGE * MERGE);
        let mut merged = zeros(tokens * MERGED)?;
        self.matrix(self.layout.merge_up, MERGED, MERGED, &norm, &mut merged);
        merged.par_chunks_mut(MERGED).for_each(|row| {
            for value in row {
                *value = gelu(*value);
            }
        });
        let mut rows = zeros(tokens * ROW)?;
        self.matrix(self.layout.merge_down, ROW, MERGED, &merged, &mut rows);
        Ok(Some(rows))
    }
}

#[inline(always)]
fn add_body(x: &[f32], y: &mut [f32]) {
    for (target, value) in y.iter_mut().zip(x) {
        *target += value;
    }
}
by_level!(add, add_wide, add_narrow, add_body, (x: &[f32], y: &mut [f32]));

fn add_all(level: Level, x: &[f32], y: &mut [f32]) {
    y.par_chunks_mut(HIDDEN * 64).zip(x.par_chunks(HIDDEN * 64)).for_each(|(y, x)| add(level, x, y));
}

/// Each patch's cosines and sines of [`ROPE`] angles: the first half by its row, the second by its column.
fn rope_angles(order: &[(u32, u32)]) -> (Vec<f32>, Vec<f32>) {
    let frequencies: Vec<f32> = (0..ROPE / 2).map(|index| (1.0 / ROPE_THETA.powf(2.0 * index as f64 / ROPE as f64)) as f32).collect();
    let mut cosines = Vec::with_capacity(order.len() * ROPE);
    let mut sines = Vec::with_capacity(order.len() * ROPE);
    for &(row, column) in order {
        for position in [row, column] {
            for frequency in &frequencies {
                let (sine, cosine) = (position as f32 * frequency).sin_cos();
                cosines.push(cosine);
                sines.push(sine);
            }
        }
    }
    (cosines, sines)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_follow_smart_resize() {
        // (height, width) -> (height, width), worked through the reference's steps
        assert_eq!(smart_resize(900, 1440).unwrap(), (896, 1440));
        assert_eq!(smart_resize(256, 256).unwrap(), (256, 256));
        // ties to even: 240 / 32 = 7.5 goes to 8, 272 / 32 = 8.5 to 8
        assert_eq!(smart_resize(240, 272).unwrap(), (256, 256));
        // too few pixels: scaled up, then each side up to a multiple
        assert_eq!(smart_resize(100, 100).unwrap(), (256, 256));
        assert_eq!(smart_resize(1, 200).unwrap(), (32, 3648));
        // too many: scaled down, then each side down to a multiple, under the limit
        let (height, width) = smart_resize(3000, 4000).unwrap();
        assert_eq!((height, width), (1760, 2336));
        assert!(height * width <= MAX_PIXELS);
        assert!(smart_resize(1, 201).is_err());
        assert!(smart_resize(0, 10).is_err());
    }

    #[test]
    fn patches_go_by_merge_block() {
        assert_eq!(merge_order(2, 4), [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3)]);
        assert_eq!(merge_order(4, 2), [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]);
    }

    /// The reference's bilinear helper over a grid of 4x2 patches, a 48-side table.
    #[test]
    fn position_taps_align_corners() {
        let close = |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).abs() < 1e-5 && (a[1] - b[1]).abs() < 1e-5;
        let (taps, weights) = axis_taps(0, 4);
        assert_eq!(taps, [0, 1]);
        assert!(close(weights, [1.0, 0.0]));
        let (taps, weights) = axis_taps(1, 4);
        assert_eq!(taps, [15, 16]);
        assert!(close(weights, [1.0 / 3.0, 2.0 / 3.0]));
        let (taps, weights) = axis_taps(2, 4);
        assert_eq!(taps, [31, 32]);
        assert!(close(weights, [2.0 / 3.0, 1.0 / 3.0]));
        let (taps, weights) = axis_taps(3, 4);
        assert_eq!(taps, [47, 47]);
        assert!(close(weights, [1.0, 0.0]));
        assert_eq!(axis_taps(1, 2).0, [47, 47]);
    }

    fn span(start: usize, grid: (usize, usize)) -> ImageSpan {
        ImageSpan { start, grid, rows: Arc::new(Vec::new()) }
    }

    /// `get_rope_index` on `text x5, image 2x3, text x4, image 4x2, text`: an image's tokens by their row and column from where
    /// the text left off, the text after it from there plus the grid's longer side.
    #[test]
    fn positions_follow_get_rope_index() {
        let spans = [span(5, (2, 3)), span(15, (4, 2))];
        assert_eq!(rope_of(&spans, 0), [0, 0, 0]);
        assert_eq!(rope_of(&spans, 4), [4, 4, 4]);
        let image: Vec<[u32; 3]> = (5..11).map(|position| rope_of(&spans, position)).collect();
        assert_eq!(image, [[5, 5, 5], [5, 5, 6], [5, 5, 7], [5, 6, 5], [5, 6, 6], [5, 6, 7]]);
        // 5 + max(2, 3) = 8
        assert_eq!(rope_of(&spans, 11), [8, 8, 8]);
        assert_eq!(rope_of(&spans, 14), [11, 11, 11]);
        let second: Vec<[u32; 3]> = (15..23).map(|position| rope_of(&spans, position)).collect();
        assert_eq!(second, [[12, 12, 12], [12, 12, 13], [12, 13, 12], [12, 13, 13], [12, 14, 12], [12, 14, 13], [12, 15, 12], [12, 15, 13]]);
        // 12 + max(4, 2) = 16
        assert_eq!(rope_of(&spans, 23), [16, 16, 16]);
        for (position, delta) in [(0, 0), (5, 0), (10, 0), (11, 3), (14, 3), (15, 3), (22, 3), (23, 7), (100, 7)] {
            assert_eq!(rope_delta(&spans, position), delta, "{position}");
            if position < 5 || (11..15).contains(&position) || position >= 23 {
                assert_eq!(rope_of(&spans, position)[0], position as u32 - delta);
            }
        }
    }

    #[test]
    fn the_exponential_and_activations_are_close() {
        let mut worst: f32 = 0.0;
        for step in 0..=20000 {
            let x = -(step as f32) * 0.004;
            let (ours, exact) = (exp_negative(x), x.exp());
            worst = worst.max((ours - exact).abs() / exact);
        }
        assert!(worst < 4e-7, "{worst}");
        // Past the point where it would underflow it is simply about the smallest normal value.
        assert!(exp_negative(-1000.0) < 1.5e-38);
        for step in -4000..=4000 {
            let x = step as f32 * 0.002;
            let inner = (2.0 / std::f64::consts::PI).sqrt() * (f64::from(x) + 0.044715 * f64::from(x).powi(3));
            let exact = 0.5 * f64::from(x) * (1.0 + inner.tanh());
            assert!((f64::from(gelu_tanh(x)) - exact).abs() < 2e-6 * (1.0 + f64::from(x).abs()), "{x}");
        }
        // erf(0.5), erf(1), erf(2) to their digits; the activation's odd points
        for (x, exact) in [(0.0, 0.0), (0.5, 0.520_499_877_8), (1.0, 0.842_700_792_9), (2.0, 0.995_322_265_0), (-1.0, -0.842_700_792_9)] {
            assert!((erf(x) - exact).abs() < 2e-7, "{x}");
        }
        assert!((gelu(1.0) - 0.841_344_7).abs() < 1e-6);
        assert!((gelu(-1.0) + 0.158_655_3).abs() < 1e-6);
    }

    /// Pseudo-random values in about [-1, 1].
    fn noise(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..count).map(|_| {
            state = state.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(0x1234_5678_9abc_def1);
            ((state >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        }).collect()
    }

    fn levels() -> Vec<Level> {
        let mut levels = vec![Level::Plain];
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                levels.push(Level::Avx2);
            }
            if level() == Level::Avx512 {
                levels.push(Level::Avx512);
            }
        }
        levels
    }

    #[test]
    fn a_dense_product_is_the_plain_sum() {
        let (rows, cols, tokens) = (131, 72, 70);
        let weights: Vec<u16> = noise(rows * cols, 3).iter().map(|value| (value.to_bits() >> 16) as u16).collect();
        let bias = noise(rows, 4);
        let x = noise(tokens * cols, 5);
        for level in levels() {
            let mut y = vec![0f32; tokens * rows];
            dense(level, &weights, rows, cols, &bias, &x, &mut y);
            for token in 0..tokens {
                for row in 0..rows {
                    let exact: f64 = (0..cols).map(|col| f64::from(widen(weights[row * cols + col])) * f64::from(x[token * cols + col])).sum::<f64>() + f64::from(bias[row]);
                    assert!((f64::from(y[token * rows + row]) - exact).abs() < 2e-4, "{level:?} {token} {row}");
                }
            }
        }
    }

    /// The tiled attention against plain scaled dot-product attention.
    #[test]
    fn attention_is_the_plain_softmax() {
        let (tokens, queries) = (1500, 3);
        let query = noise(queries * QKV, 7);
        let keys_rows = noise(tokens * HEAD, 8);
        let values = noise(tokens * HEAD, 9);
        let mut keys = vec![0f32; keys_floats(tokens)];
        for token in 0..tokens {
            for dimension in 0..HEAD {
                keys[token / KEY_TILE * HEAD * KEY_TILE + dimension * KEY_TILE + token % KEY_TILE] = keys_rows[token * HEAD + dimension];
            }
        }
        for level in levels() {
            let mut scores = vec![0f32; tokens.div_ceil(KEY_TILE) * QUERY_TILE * KEY_TILE];
            let mut out = vec![0f32; queries * HEAD];
            attend(level, &query[..(queries - 1) * QKV + HEAD], QKV, &keys, &values, tokens, &mut scores, &mut out);
            for q in 0..queries {
                let scale = 1.0 / (HEAD as f64).sqrt();
                let products: Vec<f64> = (0..tokens).map(|token| {
                    (0..HEAD).map(|d| f64::from(query[q * QKV + d]) * f64::from(keys_rows[token * HEAD + d])).sum::<f64>() * scale
                }).collect();
                let peak = products.iter().copied().fold(f64::MIN, f64::max);
                let exponentials: Vec<f64> = products.iter().map(|value| (value - peak).exp()).collect();
                let total: f64 = exponentials.iter().sum();
                for d in 0..HEAD {
                    let exact: f64 = (0..tokens).map(|token| exponentials[token] / total * f64::from(values[token * HEAD + d])).sum();
                    assert!((f64::from(out[q * HEAD + d]) - exact).abs() < 2e-5, "{level:?} {q} {d}: {} {exact}", out[q * HEAD + d]);
                }
            }
        }
    }

    /// The reference's rotate-half form, on a head's values.
    #[test]
    fn rope_turns_each_half_against_the_other() {
        let original = noise(2 * HEAD, 11);
        let (cosines, sines) = rope_angles(&[(3, 5)]);
        assert_eq!(cosines.len(), ROPE);
        for level in levels() {
            let mut turned = original.clone();
            rope(level, &mut turned, &cosines, &sines);
            for head in 0..2 {
                for index in 0..HEAD {
                    let (half, other) = if index < ROPE { (index, index + ROPE) } else { (index - ROPE, index - ROPE) };
                    let frequency = 1.0 / 10000f64.powf((half % (ROPE / 2)) as f64 * 2.0 / ROPE as f64);
                    let angle = (if half < ROPE / 2 { 3.0 } else { 5.0 }) * frequency;
                    let at = head * HEAD;
                    // q * cos + rotate_half(q) * sin, rotate_half(q) = (-q[half..], q[..half])
                    let exact = if index < ROPE {
                        f64::from(original[at + index]) * angle.cos() - f64::from(original[at + other]) * angle.sin()
                    } else {
                        f64::from(original[at + index]) * angle.cos() + f64::from(original[at + other]) * angle.sin()
                    };
                    assert!((f64::from(turned[at + index]) - exact).abs() < 1e-5, "{level:?} {head} {index}");
                }
            }
        }
    }

    #[test]
    fn layer_norm_is_torchs() {
        let x = noise(HIDDEN, 13);
        let (weight, bias) = (noise(HIDDEN, 14), noise(HIDDEN, 15));
        let mean: f64 = x.iter().map(|value| f64::from(*value)).sum::<f64>() / HIDDEN as f64;
        let variance: f64 = x.iter().map(|value| (f64::from(*value) - mean).powi(2)).sum::<f64>() / HIDDEN as f64;
        for level in levels() {
            let mut y = vec![0f32; HIDDEN];
            layer_norm(level, &x, &weight, &bias, &mut y);
            for index in 0..HIDDEN {
                let exact = (f64::from(x[index]) - mean) / (variance + 1e-6).sqrt() * f64::from(weight[index]) + f64::from(bias[index]);
                assert!((f64::from(y[index]) - exact).abs() < 1e-4, "{level:?} {index}");
            }
        }
    }

    /// The real projector's header (STRAYDOG_QWEN_MMPROJ=<file>), without its bytes.
    #[test]
    #[ignore]
    fn the_projector_has_the_layout() {
        let path = std::env::var("STRAYDOG_QWEN_MMPROJ").expect("STRAYDOG_QWEN_MMPROJ");
        let file = File::open(path).unwrap();
        let header = straydog_gguf::inspect(&mut std::io::BufReader::new(&file)).unwrap();
        let layout = layout(&header).unwrap();
        assert_eq!(layout.blocks.len(), DEPTH);
    }
}
