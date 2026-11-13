//! A partially resident layer's router in one kernel (gpu/hybrid_moe.hpp hybrid_router: the router's GEMV, the shared expert's gate, the top-10 with its
//! softmax and the export to the host) against the separate kernels it replaces (a GEMV launch a format, resident_route, hybrid_export):
//! - random routers and hidden states of Qwen3.8's shapes, for BF16 and F32 routers and gates, one to four rows, 512 experts and fewer: every
//!   score, route, weight and exported value the same bits, the one kernel launched again and again as a captured graph replays it;
//! - ties (equal scores across waves and blocks) and scores of very different sizes: the lower expert wins a tie, as the host's sort says;
//! - the export's marks of a route the GPU has resident, assists (the first routes of the absent ones, up to the assist level) or leaves to the CPU;
//! - the routers of the model in STRAYDOG_MODEL, where it is set.
//! Each case also checks the separate kernels' routes against a host's reading of the scores, so two kernels that agree on a wrong answer fail.
//!
//! cargo test --release -p straydog-engine --features cuda --test router_gpu, under the GPU lock (the real routers: STRAYDOG_MODEL naming the
//! UD-IQ1_S bundle).
#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;

use straydog_engine::gpu::{Backend, RouteCheck, ROUTE_ROW_FLOATS};

struct Random(u64);
impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Values in [-scale, scale).
    fn floats(&mut self, count: usize, scale: f32) -> Vec<f32> {
        (0..count).map(|_| ((self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale).collect()
    }
}

/// BF16 values (the bits above the low sixteen) of `values`, as a router's bytes hold them.
fn bf16(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes()).collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn device() -> Option<Backend> {
    Backend::new(0, 1 << 30).ok()
}

/// One weight set in the bytes the engine holds it in: `kind` 30 (BF16) or 0 (F32).
fn weights(values: &[f32], kind: u32) -> Vec<u8> {
    if kind == 30 { bf16(values) } else { f32_bytes(values) }
}

/// What the two kernels did with `router` and `gate` on `input`: bits against bits, then the separate kernels against a host's reading.
fn compare(label: &str, check: &RouteCheck, input: &[f32], experts: usize, assist: usize) {
    // (the one kernel is the CUDA build's; a HIP build runs the separate kernels alone, and its second copies are zeros: there the
    // separate kernels are held against the host's reading below, and nothing against the one kernel)
    assert_eq!(check.fused, cfg!(feature = "cuda"), "{label}: the one kernel took these shapes where the build has it");
    let rows = input.len() / 2560;
    let (scores_n, selected_n, probabilities_n, exported_n) = (rows * (experts + 1), rows * 10, rows * 11, rows * ROUTE_ROW_FLOATS);
    let bits = |values: &[f32]| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    if check.fused {
        assert_eq!(bits(&check.scores[..scores_n]), bits(&check.scores[scores_n..]), "{label}: scores");
        assert_eq!(check.selected[..selected_n], check.selected[selected_n..], "{label}: selected experts");
        assert_eq!(bits(&check.probabilities[..probabilities_n]), bits(&check.probabilities[probabilities_n..]), "{label}: route weights");
        assert_eq!(bits(&check.exported[..exported_n]), bits(&check.exported[exported_n..]), "{label}: exported rows");
    }

    for row in 0..rows {
        let scores = &check.scores[row * (experts + 1)..][..experts];
        let mut order: Vec<usize> = (0..experts).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap().then(a.cmp(&b)));
        let routes: Vec<i32> = (0..10).map(|k| if k < experts { order[k] as i32 } else { k as i32 }).collect();
        let selected = &check.selected[row * 10..][..10];
        assert_eq!(selected, &routes[..], "{label}: row {row}'s routes against the host's sort");
        let top = scores[order[0]];
        let weights: Vec<f32> = order[..10].iter().map(|&expert| (scores[expert] - top).exp()).collect();
        let sum: f32 = weights.iter().sum();
        let probabilities = &check.probabilities[row * 11..][..11];
        for k in 0..10 {
            let want = weights[k] / sum;
            assert!((probabilities[k] - want).abs() <= 2e-6 * want.max(1e-3), "{label}: row {row} weight {k}: {} against {want}", probabilities[k]);
        }
        let mix = check.scores[row * (experts + 1) + experts];
        let gate = 1.0 / (1.0 + (-mix).exp());
        assert!((probabilities[10] - gate).abs() <= 2e-6 * gate.max(1e-3), "{label}: row {row} gate: {} against {gate}", probabilities[10]);

        // (the exported row: routes, weights, the activation and whether the GPU takes each route: 1 a resident one (a third of the experts are
        // resident in the check), 2 one of the first `share` absent routes the GPU assists, 0 the CPU's: gpu/resident_moe.hpp expert_source)
        let exported = &check.exported[row * ROUTE_ROW_FLOATS..][..ROUTE_ROW_FLOATS];
        let absent = selected.iter().filter(|expert| **expert % 3 != 0).count();
        let share = assist.min((5 * absent + 4) / 12);
        let mut earlier = 0;
        for k in 0..10 {
            assert_eq!(exported[k], selected[k] as f32, "{label}: row {row} exported route {k}");
            let wanted = if selected[k] % 3 == 0 {
                1.0
            } else {
                earlier += 1;
                if earlier <= share { 2.0 } else { 0.0 }
            };
            assert_eq!(exported[2592 + k], wanted, "{label}: row {row} exported source {k}");
        }
        assert_eq!(bits(&exported[16..27]), bits(probabilities), "{label}: row {row} exported weights");
        assert_eq!(bits(&exported[32..2592]), bits(&input[row * 2560..][..2560]), "{label}: row {row} exported activation");
        assert!(exported[10..16].iter().chain(&exported[27..32]).chain(&exported[2602..]).all(|value| *value == 0.0), "{label}: row {row} exported gaps");
    }
}

fn run(gpu: &mut Backend, label: &str, router: (&[f32], u32), gate: (&[f32], u32), experts: usize, input: &[f32], repeats: u32) {
    run_assisted(gpu, label, router, gate, experts, input, repeats, 0);
}

#[allow(clippy::too_many_arguments)]
fn run_assisted(gpu: &mut Backend, label: &str, router: (&[f32], u32), gate: (&[f32], u32), experts: usize, input: &[f32], repeats: u32,
                assist: u32) {
    let (router_bytes, gate_bytes) = (weights(router.0, router.1), weights(gate.0, gate.1));
    let check = gpu.route_check((&router_bytes, router.1), (&gate_bytes, gate.1), experts as u64, input, repeats, assist).unwrap();
    compare(label, &check, input, experts, assist as usize);
}

#[test]
fn the_one_kernel_routes_as_the_separate_kernels_do() {
    let Some(mut gpu) = device() else { return };
    let mut random = Random(0x9e37_79b9_7f4a_7c15);
    // Qwen3.8's layer: 512 experts, BF16 router, and the gate as the bundle holds it (F32, or BF16 where its values allow)
    for (router_kind, gate_kind) in [(30, 0), (30, 30), (0, 0), (0, 30)] {
        for rows in 1..=4 {
            let router = random.floats(512 * 2560, 0.05);
            let gate = random.floats(2560, 0.05);
            let input = random.floats(rows * 2560, 2.0);
            run(&mut gpu, &format!("512 experts, router {router_kind} gate {gate_kind}, {rows} rows"), (&router, router_kind), (&gate, gate_kind), 512, &input, 4);
        }
    }
    // fewer experts: a top-10 over 10, over a partial wave, over a few waves
    for experts in [10, 11, 37, 64, 65, 200, 256, 511] {
        let router = random.floats(experts * 2560, 0.05);
        let gate = random.floats(2560, 0.05);
        let input = random.floats(2 * 2560, 1.0);
        run(&mut gpu, &format!("{experts} experts"), (&router, 30), (&gate, 0), experts, &input, 3);
    }
}

#[test]
fn the_exported_sources_follow_the_assist() {
    let Some(mut gpu) = device() else { return };
    let mut random = Random(99);
    for assist in [1, 3, 5, 10] {
        for rows in [1, 4] {
            let router = random.floats(512 * 2560, 0.05);
            let gate = random.floats(2560, 0.05);
            let input = random.floats(rows * 2560, 2.0);
            run_assisted(&mut gpu, &format!("assist {assist}, {rows} rows"), (&router, 30), (&gate, 0), 512, &input, 3, assist);
        }
    }
}

#[test]
fn ties_and_scores_of_many_sizes() {
    let Some(mut gpu) = device() else { return };
    let mut random = Random(42);
    // every expert's row one of seven: scores tie in sevens, across waves and blocks, and the lower expert wins
    let seven = random.floats(7 * 2560, 0.05);
    let tied: Vec<f32> = (0..512).flat_map(|expert| seven[(expert % 7) * 2560..][..2560].to_vec()).collect();
    let gate = random.floats(2560, 0.05);
    for rows in [1, 3] {
        let input = random.floats(rows * 2560, 1.0);
        run(&mut gpu, &format!("ties in sevens, {rows} rows"), (&tied, 30), (&gate, 0), 512, &input, 3);
    }
    // a router of zeros: every score 0, the routes are the first ten experts
    let zeros = vec![0.0f32; 512 * 2560];
    let input = random.floats(2560, 1.0);
    run(&mut gpu, "zeros", (&zeros, 30), (&gate, 30), 512, &input, 2);
    // large scores (the softmax underflows past the first few), and an input that is zero
    let big = random.floats(512 * 2560, 4.0);
    let input = random.floats(2 * 2560, 8.0);
    run(&mut gpu, "large scores", (&big, 30), (&gate, 0), 512, &input, 2);
    run(&mut gpu, "zero input", (&big, 30), (&gate, 0), 512, &vec![0.0; 2560], 2);
}

/// The routers of the model in STRAYDOG_MODEL (a few layers of the UD-IQ1_S bundle), on random hidden states.
#[test]
fn the_models_routers() {
    let Some(root) = common::model_env("STRAYDOG_MODEL", "the UD-IQ1_S model bundle directory") else { return };
    let Some(mut gpu) = device() else { return };
    let id = straydog_engine::bundle::Id::find(Some(straydog_engine::bundle::MODEL), None, "UD-IQ1_S").unwrap();
    let weights = straydog_engine::weights::Weights::open(&std::path::PathBuf::from(root), id, None, 0, &|_, _, _| {}).unwrap();
    let mut random = Random(7);
    for layer in [0, 1, 11, 24, 39, 47] {
        let held = |name: &str| {
            let (bytes, kind) = weights.raw(&format!("blk.{layer}.{name}"), 0, None).unwrap();
            // (the engine holds F32 values that are BF16 values as BF16: Model::router, Model::mix_row)
            match kind {
                0 if straydog_engine::gpu::bf16_exact(bytes) => (straydog_engine::gpu::narrow_bf16(bytes, name).unwrap(), 30),
                0 | 30 => (bytes.to_vec(), kind),
                other => panic!("blk.{layer}.{name} is of kind {other}"),
            }
        };
        let (router, router_kind) = held("ffn_gate_inp.weight");
        let (gate, gate_kind) = held("ffn_gate_inp_shexp.weight");
        // (a partially resident layer holds the gate as an F32 vector, Model::vector; a fully resident one as `gate` above, Model::mix_row)
        let vector: Vec<u8> = if gate_kind == 30 {
            gate.chunks_exact(2).flat_map(|value| [0, 0, value[0], value[1]]).collect()
        } else {
            gate.clone()
        };
        eprintln!("blk.{layer}: router kind {router_kind}, gate kind {gate_kind}");
        for rows in [1, 2] {
            let input = random.floats(rows * 2560, 1.5);
            for (held_gate, kind, how) in [(&vector, 0, "partially resident"), (&gate, gate_kind, "fully resident")] {
                let check = gpu.route_check((&router, router_kind), (held_gate, kind), 512, &input, 4, 0).unwrap();
                compare(&format!("blk.{layer}, {how}, {rows} rows"), &check, &input, 512, 0);
            }
        }
    }
}
