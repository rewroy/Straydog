//! Drafts for speculative decoding from the sequence itself: where the last few tokens occurred
//! before, the tokens that followed them then are proposed to follow now (prompt lookup). Code
//! being edited, a document quoted back or a tool's output repeated make long matches; prose rarely
//! does, and without a match nothing is proposed, so a reply that repeats nothing is decoded as
//! without drafts.

use std::collections::HashMap;

/// The fewest tokens a match takes before it proposes anything.
const SHORTEST: usize = 3;
/// How far back a match is measured.
const LONGEST: usize = 32;

/// A match of this many tokens (the policy's widest) is verified without asking a drafter.
pub const LONG_MATCH: usize = 8;

/// The sequence's tokens and, for every three in a row, where they last ended and where before.
#[derive(Default)]
pub struct Ngrams {
    tokens: Vec<u32>,
    ends: HashMap<u64, (u32, Option<u32>)>,
}

fn key(three: &[u32]) -> u64 {
    // (a vocabulary below 2^21 keeps the three apart)
    (u64::from(three[0]) << 42) ^ (u64::from(three[1]) << 21) ^ u64::from(three[2])
}

impl Ngrams {
    pub fn new() -> Ngrams {
        Ngrams::default()
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn push(&mut self, token: u32) {
        self.tokens.push(token);
        let end = self.tokens.len();
        if end >= SHORTEST {
            let at = (end - 1) as u32;
            self.ends.entry(key(&self.tokens[end - SHORTEST..])).and_modify(|(last, before)| *before = Some(std::mem::replace(last, at)))
                .or_insert((at, None));
        }
    }

    pub fn extend(&mut self, tokens: &[u32]) {
        for &token in tokens {
            self.push(token);
        }
    }

    /// Drop the tokens after the first `keep` (as the sequence was then).
    pub fn truncate(&mut self, keep: usize) {
        if keep >= self.tokens.len() {
            return;
        }
        let tokens = std::mem::take(&mut self.tokens);
        self.ends.clear();
        self.extend(&tokens[..keep]);
    }

    /// The tokens that followed the latest earlier occurrence of the sequence's last tokens (at
    /// most `most`), and how many tokens that occurrence matched; none below three.
    pub fn propose(&self, most: usize) -> (Vec<u32>, usize) {
        let end = self.tokens.len();
        if most == 0 || end < SHORTEST + 1 {
            return (Vec::new(), 0);
        }
        let Some(&(last, before)) = self.ends.get(&key(&self.tokens[end - SHORTEST..])) else { return (Vec::new(), 0) };
        // (the latest end is the sequence's own unless a later push moved it)
        let earlier = if last as usize == end - 1 { before } else { Some(last) };
        let Some(earlier) = earlier.map(|at| at as usize) else { return (Vec::new(), 0) };
        let matched = (0..LONGEST.min(earlier + 1)).take_while(|&back| self.tokens[earlier - back] == self.tokens[end - 1 - back]).count();
        if matched < SHORTEST {
            return (Vec::new(), 0);
        }
        let drafts = self.tokens[earlier + 1..].iter().take(most).copied().collect();
        (drafts, matched)
    }
}

/// How many drafts a verification pass takes: more for longer matches, none while recent ones
/// were mostly rejected (a pass of several rows costs more than one, so drafts pay only when most
/// are accepted).
pub struct Policy {
    /// The most tokens a verification takes (the engines' verify_rows), the fed token among them.
    rows: usize,
    /// Accepted drafts over proposed ones, recently (an exponential average).
    rate: f64,
    /// Passes left without drafts after the rate fell low.
    resting: usize,
}

/// Below this acceptance rate, drafts rest for [`REST`] passes.
const LOW_RATE: f64 = 0.35;
const REST: usize = 32;

impl Policy {
    pub fn new(rows: usize) -> Policy {
        Policy { rows, rate: 0.6, resting: 0 }
    }

    /// How many of a match's drafts to verify, given how many tokens it matched.
    pub fn width(&mut self, matched: usize) -> usize {
        if self.rows < 2 || matched < SHORTEST {
            return 0;
        }
        if self.resting > 0 {
            self.resting -= 1;
            return 0;
        }
        let wanted = match matched {
            0..=3 => 1,
            4..=7 => 2,
            _ => 3,
        };
        wanted.min(self.rows - 1)
    }

    /// A pass verified `proposed` drafts and accepted `accepted`.
    pub fn record(&mut self, proposed: usize, accepted: usize) {
        if proposed == 0 {
            return;
        }
        self.rate = 0.8 * self.rate + 0.2 * accepted as f64 / proposed as f64;
        if self.rate < LOW_RATE {
            self.resting = REST;
            self.rate = 0.5;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_repeated_run_proposes_what_followed_it() {
        let mut ngrams = Ngrams::new();
        ngrams.extend(&[1, 2, 3, 4, 5, 6, 9, 9, 1, 2, 3]);
        assert_eq!(ngrams.propose(3), (vec![4, 5, 6], 3));
        ngrams.push(4);
        assert_eq!(ngrams.propose(2), (vec![5, 6], 4));
        // the latest earlier occurrence wins
        ngrams.extend(&[7, 2, 3, 4]);
        assert_eq!(ngrams.propose(1), (vec![7], 3));
    }

    #[test]
    fn nothing_is_proposed_without_a_match_of_three() {
        let mut ngrams = Ngrams::new();
        ngrams.extend(&[1, 2, 3, 4, 1, 2]);
        assert_eq!(ngrams.propose(3), (vec![], 0));
        ngrams.extend(&[8]);
        assert_eq!(ngrams.propose(3), (vec![], 0));
        assert_eq!(Ngrams::new().propose(3), (vec![], 0));
    }

    #[test]
    fn a_match_running_into_itself_proposes_the_repetition() {
        let mut ngrams = Ngrams::new();
        ngrams.extend(&[5, 5, 5, 5]);
        assert_eq!(ngrams.propose(3), (vec![5], 3));
    }

    #[test]
    fn truncating_forgets_the_dropped_tokens() {
        let mut ngrams = Ngrams::new();
        ngrams.extend(&[1, 2, 3, 4, 1, 2, 3]);
        ngrams.truncate(4);
        ngrams.extend(&[9, 2, 3]);
        assert_eq!(ngrams.propose(3), (vec![], 0));
        assert_eq!(ngrams.len(), 7);
    }

    #[test]
    fn the_policy_widens_with_the_match_and_rests_after_rejections() {
        let mut policy = Policy::new(4);
        assert_eq!((policy.width(2), policy.width(3), policy.width(5), policy.width(12)), (0, 1, 2, 3));
        for _ in 0..8 {
            policy.record(3, 0);
        }
        assert_eq!(policy.width(12), 0);
        assert_eq!(Policy::new(1).width(12), 0);
        assert_eq!(Policy::new(2).width(12), 1);
    }
}
