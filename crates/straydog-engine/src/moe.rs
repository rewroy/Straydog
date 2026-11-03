//! Routed experts over many rows at once (a long prefill), expert by expert: the routes grouped by
//! expert, waves of experts to make resident at a time, and an expert's two stages on a device
//! (gather its rows, gate and up, SwiGLU times the routes' weights; then down, added to the rows'
//! output). Each expert's weights are read once for all its rows. A device may run only its share
//! of an expert (tensor parallel): its rows of gate and up into its columns of the activations, its
//! rows of down into its columns of the output; the other columns of the buffers it writes stay
//! zero, so the devices' activations and outputs add up across devices.

use std::collections::BTreeMap;
use std::ffi::c_void;

use crate::gpu::{Backend, Result};
use crate::ops::Buffer;

type P = *const c_void;
type M = *mut c_void;

/// The routes of a pass grouped by expert: each route's row and weight, an expert's together in
/// row order, the experts in ascending order.
pub struct Grouped {
    pub rows: Vec<i32>,
    pub weights: Vec<f32>,
    pub spans: Vec<Span>,
}

/// An expert's routes in [`Grouped`]: `routes` of them from `first`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub expert: usize,
    pub first: usize,
    pub routes: usize,
}

impl Grouped {
    /// From each row's `k` routes (`ids`, `weights`: row-major).
    pub fn new(ids: &[i32], weights: &[f32], k: usize) -> Grouped {
        let mut by_expert: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (route, &expert) in ids.iter().enumerate() {
            by_expert.entry(expert as usize).or_default().push(route);
        }
        let mut grouped = Grouped { rows: Vec::with_capacity(ids.len()), weights: Vec::with_capacity(ids.len()), spans: Vec::new() };
        for (expert, routes) in by_expert {
            grouped.spans.push(Span { expert, first: grouped.rows.len(), routes: routes.len() });
            grouped.rows.extend(routes.iter().map(|&route| (route / k) as i32));
            grouped.weights.extend(routes.iter().map(|&route| weights[route]));
        }
        grouped
    }

    /// The experts in waves of at most `experts` of them and `routes` routes (an expert of more
    /// routes alone).
    pub fn waves(&self, experts: usize, routes: usize) -> Vec<&[Span]> {
        let (mut waves, mut start) = (Vec::new(), 0);
        while start < self.spans.len() {
            let (mut end, mut taken) = (start, 0);
            while end < self.spans.len() && (end == start || (end - start < experts && taken + self.spans[end].routes <= routes)) {
                taken += self.spans[end].routes;
                end += 1;
            }
            waves.push(&self.spans[start..end]);
            start = end;
        }
        waves
    }

    /// The `waves`' routes by row, for adding a wave's outputs to `rows` rows in one pass (see
    /// `Backend::scatter_rows`): wave i's row offsets at `i * (rows + 1)`, `rows + 1` of them,
    /// indexing this table; at those positions, from `waves.len() * (rows + 1)` on, each row's routes
    /// as the wave numbers them (its first route 0), in the wave's order (expert by expert), which is
    /// the order an expert at a time adds them in.
    pub fn by_row(&self, waves: &[&[Span]], rows: usize) -> Vec<i32> {
        let lists = waves.len() * (rows + 1);
        let routes: usize = waves.iter().flat_map(|wave| wave.iter()).map(|span| span.routes).sum();
        let mut table = vec![0i32; lists + routes];
        let mut at = lists;
        for (index, wave) in waves.iter().enumerate() {
            let Some(base) = wave.first().map(|span| span.first) else { continue };
            let count: usize = wave.iter().map(|span| span.routes).sum();
            let own = &self.rows[base..base + count];
            let offsets = &mut table[index * (rows + 1)..(index + 1) * (rows + 1)];
            for &row in own {
                offsets[row as usize + 1] += 1;
            }
            offsets[0] = at as i32;
            for row in 0..rows {
                offsets[row + 1] += offsets[row];
            }
            let mut next: Vec<i32> = offsets[..rows].to_vec();
            for (route, &row) in own.iter().enumerate() {
                table[next[row as usize] as usize] = route as i32;
                next[row as usize] += 1;
            }
            at += count;
        }
        table
    }
}

/// An expert as a slot holds it: gate and up (`gate_bytes` each) and down (`down_bytes`) back to
/// back, of `kinds`; `width` and `ffn` its input and hidden widths, `ffn_rows` and `hidden_rows` a
/// device's share of them (first, count): of gate and up its rows, of down its rows.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub gate_bytes: u64,
    pub down_bytes: u64,
    pub kinds: [u32; 3],
    pub width: usize,
    pub ffn: usize,
    pub ffn_rows: (usize, usize),
    pub hidden_rows: (usize, usize),
}

/// Buffers for [`gate_up`] and [`down`]: the linears' scratch (weights expanded for a GEMM), an
/// expert's rows of the input (`width` each), its gate and up (`ffn` each, the device's columns
/// written), its output (`width`, the device's columns).
pub struct Scratch<'a> {
    pub gemm: &'a Buffer,
    /// Only the expert's large-row linears; every non-GEMM op remains F32.
    pub half_gemm: bool,
    pub input: &'a Buffer,
    pub gate: &'a Buffer,
    pub up: &'a Buffer,
    pub output: &'a Buffer,
}

/// Stage 1 of an expert over its `count` routes: their rows of `x` (of `x_rows`; `rows` and
/// `weights` the routes' rows and weights on the device) through gate and up, clamped SwiGLU times
/// the routes' weights into `act` (`count` rows of `ffn`; the device's columns, the others zero).
#[allow(clippy::too_many_arguments)]
pub fn gate_up(gpu: &mut Backend, slot: &Buffer, shape: &Shape, scratch: &Scratch<'_>, x: P, x_rows: usize, rows: P, weights: P,
               count: usize, clamp: f32, act: M) -> Result<()> {
    let (width, ffn, (first, share)) = (shape.width as u64, shape.ffn as u64, shape.ffn_rows);
    gpu.gather(x, x_rows as u64, rows, count as u64, width, scratch.input.base())?;
    gpu.linear_gemm(&slot.slice(0, shape.gate_bytes), shape.kinds[0], share as u64, width, 0, scratch.input.base(), 0, count as u64,
                    scratch.gate.at(first), ffn, scratch.gemm, scratch.half_gemm)?;
    gpu.linear_gemm(&slot.slice(shape.gate_bytes, shape.gate_bytes), shape.kinds[1], share as u64, width, 0, scratch.input.base(), 0,
                    count as u64, scratch.up.at(first), ffn, scratch.gemm, scratch.half_gemm)?;
    gpu.swiglu(scratch.gate.base(), scratch.up.base(), count as u64, ffn, clamp, Some(weights), act)
}

/// Stage 2 of an expert over its `count` routes: their activations (`act`, rows of `ffn`, every
/// device's columns) through down, the device's columns added to the routes' rows of `out` (of
/// `out_rows`, rows of `width`).
#[allow(clippy::too_many_arguments)]
pub fn down(gpu: &mut Backend, slot: &Buffer, shape: &Shape, scratch: &Scratch<'_>, act: P, count: usize, rows: P, out: M,
            out_rows: usize) -> Result<()> {
    let (width, (first, share)) = (shape.width as u64, shape.hidden_rows);
    gpu.linear_gemm(&slot.slice(2 * shape.gate_bytes, shape.down_bytes), shape.kinds[2], share as u64, shape.ffn as u64, 0, act, 0,
                    count as u64, scratch.output.at(first), width, scratch.gemm, scratch.half_gemm)?;
    gpu.scatter_add(scratch.output.base(), rows, count as u64, width, out, out_rows as u64)
}

#[cfg(test)]
mod tests {
    use super::{Grouped, Span};

    #[test]
    fn routes_group_by_expert_in_row_order_and_waves_fit() {
        // three rows, two routes each
        let grouped = Grouped::new(&[5, 1, 1, 7, 5, 1], &[0.5, 0.25, 0.75, 0.125, 1.0, 2.0], 2);
        assert_eq!(grouped.rows, [0, 1, 2, 0, 2, 1]);
        assert_eq!(grouped.weights, [0.25, 0.75, 2.0, 0.5, 1.0, 0.125]);
        assert_eq!(grouped.spans, [Span { expert: 1, first: 0, routes: 3 }, Span { expert: 5, first: 3, routes: 2 },
                                   Span { expert: 7, first: 5, routes: 1 }]);
        let waves = grouped.waves(2, 4);
        assert_eq!(waves.iter().map(|wave| wave.len()).collect::<Vec<_>>(), [1, 2]);
        assert_eq!(grouped.waves(2, 1).iter().map(|wave| wave.len()).collect::<Vec<_>>(), [1, 1, 1]);
    }

    #[test]
    fn a_wave_s_routes_by_row_come_in_the_wave_s_order() {
        // rows 0..3, routes grouped [0, 1, 2 | 0, 2 | 1]; waves [expert 1] and [experts 5, 7]
        let grouped = Grouped::new(&[5, 1, 1, 7, 5, 1], &[0.5, 0.25, 0.75, 0.125, 1.0, 2.0], 2);
        let waves = grouped.waves(2, 4);
        let table = grouped.by_row(&waves, 3);
        let lists = 2 * 4;
        // wave 0: rows 0, 1, 2 one route each (0, 1, 2)
        assert_eq!(&table[..4], &[lists as i32, lists as i32 + 1, lists as i32 + 2, lists as i32 + 3]);
        assert_eq!(&table[lists..lists + 3], &[0, 1, 2]);
        // wave 1 (its routes: expert 5's rows 0 and 2, then expert 7's row 1): row 0 route 0, row 1 route 2, row 2 route 1
        let w1 = lists as i32 + 3;
        assert_eq!(&table[4..8], &[w1, w1 + 1, w1 + 2, w1 + 3]);
        assert_eq!(&table[lists + 3..], &[0, 2, 1]);
        // a row of two routes in one wave keeps the wave's order
        let both = Grouped::new(&[3, 4, 4, 3], &[1.0; 4], 2);
        let waves = both.waves(2, 8);
        assert_eq!(waves.len(), 1);
        let table = both.by_row(&waves, 2);
        assert_eq!(&table[..3], &[3, 5, 7]);
        // expert 3's routes (rows 0, 1) are the wave's 0 and 1, expert 4's (rows 0, 1) its 2 and 3
        assert_eq!(&table[3..], &[0, 2, 1, 3]);
    }
}
