//! CPU-only costs of `suppress` on a model's vocabulary, with synthetic logits: the table, a row's mask and a token's choice.
//! cargo run --release -p straydog-engine --example suppress_cost -- <tokenizer.json> [samples=2001]
//! No GPU features/model. Microseconds, the median and the 90th percentile.
use std::{hint::black_box, sync::Arc, time::Instant};
use straydog_engine::{
    constraint::{Suppression, TokenBytes},
    model::Rng,
    sampling::{self, Generated, Sampling},
    text::Text,
};

/// Qwen3.8's head.
const ROW: usize = 248_320;

fn measure(name: &str, samples: usize, mut f: impl FnMut()) {
    let mut times: Vec<f64> = (0..samples).map(|_| {
        let start = Instant::now();
        f();
        start.elapsed().as_secs_f64() * 1e6
    }).collect();
    times.sort_by(f64::total_cmp);
    println!("{name}: {:.2} us, p90 {:.2} us", times[samples / 2], times[samples * 9 / 10]);
}

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    let text = Text::new(&std::fs::read(&arguments[1]).expect("tokenizer.json")).expect("a tokenizer");
    let samples = arguments.get(2).map_or(2001, |count| count.parse().expect("a count"));
    let start = Instant::now();
    let tokens: Arc<TokenBytes> = text.token_bytes();
    println!("token bytes (every request reads them already): {:.2} ms", start.elapsed().as_secs_f64() * 1e3);
    let start = Instant::now();
    let (always, openers) = tokens.suppressed().counts();
    println!("the table, once: {:.2} ms; {always} tokens always refused, {openers} that may finish a letter", start.elapsed().as_secs_f64() * 1e3);

    let mut rng = Rng::new(1);
    let logits: Vec<f32> = (0..ROW).map(|_| (rng.random() as f32 - 0.5) * 16.0).collect();
    let suppression = Suppression::new(Arc::clone(&tokens));

    let mut row = logits.clone();
    measure("a row's mask", samples, || suppression.mask(black_box(&mut row)));
    let mut open = suppression.clone();
    // ("\xE5\x91", the first bytes of a letter split across tokens, if the vocabulary has them)
    let lead = (0..tokens.len() as u32).find(|&token| tokens.text_of(token) == Some(&b"\xE5\x91"[..]));
    if let Some(lead) = lead {
        open.advance(lead);
        measure("a row's mask, a letter left open", samples, || open.mask(black_box(&mut row)));
    }
    measure("whether a token is allowed", samples, || { black_box(suppression.allows(black_box(12345))); });
    measure("a token taken", samples, || black_box(&mut open).advance(black_box(12345)));

    let generated = Generated::new(ROW);
    for (name, sampling) in [("card", Sampling::recommended(true)), ("plain", Sampling { temperature: 1.0, ..Sampling::GREEDY }), ("greedy", Sampling::GREEDY)] {
        let mut rng = Rng::new(2);
        measure(&format!("{name}: a token without it"), samples.min(301), || {
            black_box(sampling::choose(black_box(&logits), &sampling, &generated, &mut rng));
        });
        let mut rng = Rng::new(2);
        measure(&format!("{name}: a token with it"), samples.min(301), || {
            black_box(sampling::choose_where(black_box(&logits), &sampling, &generated, &mut rng, &mut |token| suppression.allows(token)));
        });
    }
    let mut masked = logits.clone();
    suppression.mask(&mut masked);
    measure("a masked row's top 20, read on the host", samples.min(301), || { black_box(sampling::top_of(black_box(&masked), 20)); });
    // A refusal: the card's draw kept to tokens that, as it happens, are all refused, then masked by the check over the row.
    let sampling = Sampling::recommended(true);
    let mut rng = Rng::new(3);
    measure("the card, every draw refused: the whole row checked", samples.min(301), || {
        let mut refused = 0;
        black_box(sampling::choose_where(black_box(&logits), &sampling, &generated, &mut rng, &mut |token| {
            refused += 1;
            refused > 64 && suppression.allows(token)
        }));
    });
}
