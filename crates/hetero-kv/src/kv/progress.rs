//! Validated, bounded-by-coverage layer progress for a reserved KV slab.
//!
//! A producer publishes `(layer_start, layer_end, token_start, token_end)` spans as it writes
//! them; a consumer drains the spans and imports them. Spans must stay inside the reservation,
//! may arrive in any order, and must never overlap. [`LayerProgress::finish`] succeeds only once
//! every layer covers exactly `start..end`.
//!
//! This tracks *source writes* only. A consumer must still fence its own copies before it
//! publishes the cache or releases the reservation: source completion is not a fence.
//!
//! Semantics and error messages match Optimus `accel/pd/layer_progress.py`. Arguments are
//! signed so negative values from foreign callers are rejected with the same messages.

use std::sync::Mutex;

use crate::error::{invalid, Result};

/// One published span: layers `layers.0..layers.1`, tokens `tokens.0..tokens.1`.
pub type ProgressSpan = (i64, i64, i64, i64);

struct Inner {
    coverage: Vec<Vec<(i64, i64)>>,
    pending: Vec<ProgressSpan>,
    closed: bool,
}

/// Coverage tracker for `layers` layers over tokens `start..end`. Thread-safe.
pub struct LayerProgress {
    layers: i64,
    start: i64,
    end: i64,
    inner: Mutex<Inner>,
}

impl LayerProgress {
    /// New tracker; `start == end` (an empty suffix) is valid and needs no spans.
    pub fn new(layers: i64, start: i64, end: i64) -> Result<Self> {
        if layers <= 0 || start < 0 || start > end {
            return invalid("invalid layer-progress geometry");
        }
        Ok(LayerProgress {
            layers,
            start,
            end,
            inner: Mutex::new(Inner {
                coverage: vec![Vec::new(); layers as usize],
                pending: Vec::new(),
                closed: false,
            }),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Layers tracked.
    pub fn layers(&self) -> i64 {
        self.layers
    }
    /// First token of the reservation.
    pub fn start(&self) -> i64 {
        self.start
    }
    /// End token of the reservation (exclusive).
    pub fn end(&self) -> i64 {
        self.end
    }

    /// Record a span. The whole span is validated before any layer's coverage changes.
    pub fn publish(&self, span: ProgressSpan) -> Result<()> {
        let (l0, l1, t0, t1) = span;
        let mut inner = self.lock();
        if inner.closed {
            return invalid("layer progress is closed");
        }
        if !(0 <= l0
            && l0 < l1
            && l1 <= self.layers
            && self.start <= t0
            && t0 < t1
            && t1 <= self.end)
        {
            return invalid("invalid layer-progress span");
        }
        let layers = l0 as usize..l1 as usize;
        if inner.coverage[layers.clone()]
            .iter()
            .any(|c| c.iter().any(|&(a, b)| t0 < b && a < t1))
        {
            return invalid("overlapping layer-progress span");
        }
        for c in &mut inner.coverage[layers] {
            c.push((t0, t1));
        }
        inner.pending.push(span);
        Ok(())
    }

    /// Take every span published since the last drain, in publication order.
    pub fn drain(&self) -> Vec<ProgressSpan> {
        std::mem::take(&mut self.lock().pending)
    }

    /// True if spans are waiting to be drained.
    pub fn pending(&self) -> bool {
        !self.lock().pending.is_empty()
    }

    /// True after a successful [`LayerProgress::finish`].
    pub fn finished(&self) -> bool {
        self.lock().closed
    }

    /// Close the tracker once every layer covers the reservation exactly.
    pub fn finish(&self) -> Result<()> {
        let mut inner = self.lock();
        if inner.closed {
            return invalid("layer progress is closed");
        }
        for intervals in &inner.coverage {
            let mut sorted = intervals.clone();
            sorted.sort_unstable();
            let mut position = self.start;
            for (a, b) in sorted {
                if a != position {
                    return invalid("incomplete layer-progress coverage");
                }
                position = b;
            }
            if position != self.end {
                return invalid("incomplete layer-progress coverage");
            }
        }
        inner.closed = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(r: Result<()>) -> String {
        r.unwrap_err().to_string()
    }

    #[test]
    fn partial_prefix_and_out_of_order_layers_require_complete_coverage() {
        let p = LayerProgress::new(3, 5, 9).unwrap();
        p.publish((1, 3, 5, 9)).unwrap();
        p.publish((0, 1, 7, 9)).unwrap();
        assert_eq!(p.drain(), vec![(1, 3, 5, 9), (0, 1, 7, 9)]);
        assert!(msg(p.finish()).contains("incomplete"));
        p.publish((0, 1, 5, 7)).unwrap();
        p.finish().unwrap();
        assert!(p.finished());
        assert_eq!(p.drain(), vec![(0, 1, 5, 7)]);
        assert!(p.drain().is_empty());
        assert!(msg(p.publish((0, 1, 5, 7))).contains("closed"));
    }

    #[test]
    fn overlap_rejection_is_atomic_across_layers() {
        let p = LayerProgress::new(3, 0, 4).unwrap();
        p.publish((1, 2, 0, 2)).unwrap();
        assert!(msg(p.publish((0, 3, 1, 3))).contains("overlap"));
        p.publish((0, 1, 0, 4)).unwrap();
        p.publish((1, 2, 2, 4)).unwrap();
        p.publish((2, 3, 0, 4)).unwrap();
        p.finish().unwrap();
        assert_eq!(p.drain().len(), 4);
    }

    #[test]
    fn invalid_spans_never_enter_the_queue() {
        for span in [
            (-1, 1, 0, 4),
            (0, 4, 0, 4),
            (0, 1, -1, 4),
            (0, 1, 0, 5),
            (0, 0, 0, 4),
            (0, 1, 2, 2),
        ] {
            let p = LayerProgress::new(3, 0, 4).unwrap();
            assert!(p.publish(span).is_err());
            assert!(!p.pending());
        }
        assert!(LayerProgress::new(0, 0, 1).is_err());
        assert!(LayerProgress::new(1, 2, 1).is_err());
    }

    #[test]
    fn empty_suffix_needs_no_transfer() {
        let p = LayerProgress::new(3, 8, 8).unwrap();
        p.finish().unwrap();
        assert!(p.drain().is_empty());
    }
}
