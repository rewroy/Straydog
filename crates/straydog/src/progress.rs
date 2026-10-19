//! A progress line for long downloads and authentications: rewritten in place a few times a
//! second, and ended when a stage completes.

use std::sync::Mutex;
use std::time::Instant;

static LAST: Mutex<Option<(String, Instant)>> = Mutex::new(None);

/// Report `done` of `total` bytes of `stage` on stderr.
pub fn report(stage: &str, done: u64, total: u64) {
    if total == 0 {
        return;
    }
    let mut last = LAST.lock().unwrap();
    let finished = done >= total;
    let changed = last.as_ref().is_none_or(|(previous, _)| previous != stage);
    if !finished && !changed && last.as_ref().is_some_and(|(_, at)| at.elapsed().as_millis() < 250) {
        return;
    }
    *last = Some((stage.to_string(), Instant::now()));
    eprint!("\r{stage}: {:.1}/{:.1} GB", done as f64 / 1e9, total as f64 / 1e9);
    if finished {
        eprintln!();
    }
}
