//! Layout -> byte-span plans.
//!
//! A [`Selection`] names *which* KV moves in logical terms: a layer range and a token range
//! addressed through block tables on each side. [`plan`] turns it into coalesced byte spans
//! for the two buffers' layouts. All of these are special cases of one selection:
//!
//! * token-major prefix copy of a contiguous cache: `Selection::prefix(layers, n)`;
//! * paged block transfer (vLLM `block_ids` -> remote `block_ids`): `Selection::blocks`;
//! * layer streaming of a token span: `Selection::contiguous(l0..l1, src_start, dst_start, n)`;
//! * paged -> contiguous gather, or any mix, by giving one side a block table.

use std::ops::Range;

use super::layout::{KvSpec, Layout, Split};
use crate::engine::Span;
use crate::error::{invalid, Result};

/// Logical KV selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    /// Layers to copy.
    pub layers: Range<u32>,
    /// Source block table: logical block `i` lives in physical block `src_blocks[i]`.
    pub src_blocks: Vec<u32>,
    /// Destination block table.
    pub dst_blocks: Vec<u32>,
    /// First logical source token.
    pub src_start: u64,
    /// First logical destination token.
    pub dst_start: u64,
    /// Number of tokens; `None` means the rest of the source block table.
    pub tokens: Option<u64>,
}

impl Selection {
    /// Whole blocks: `src_ids[i]` -> `dst_ids[i]`.
    pub fn blocks(layers: Range<u32>, src_ids: Vec<u32>, dst_ids: Vec<u32>) -> Selection {
        Selection {
            layers,
            src_blocks: src_ids,
            dst_blocks: dst_ids,
            src_start: 0,
            dst_start: 0,
            tokens: None,
        }
    }

    /// Token range of contiguous (single-block) caches.
    pub fn contiguous(
        layers: Range<u32>,
        src_start: u64,
        dst_start: u64,
        tokens: u64,
    ) -> Selection {
        Selection {
            layers,
            src_blocks: vec![0],
            dst_blocks: vec![0],
            src_start,
            dst_start,
            tokens: Some(tokens),
        }
    }

    /// The first `tokens` tokens of contiguous caches.
    pub fn prefix(layers: Range<u32>, tokens: u64) -> Selection {
        Self::contiguous(layers, 0, 0, tokens)
    }
}

/// Coalesced byte spans for one selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    spans: Vec<Span>,
    bytes: u64,
    layers: Range<u32>,
}

impl Plan {
    /// Spans, indexed against the source and destination buffers' region lists.
    pub fn spans(&self) -> &[Span] {
        &self.spans
    }
    /// Total bytes.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    /// Layers covered (for per-layer completion tracking).
    pub fn layers(&self) -> Range<u32> {
        self.layers.clone()
    }
}

/// Map a logical token to (physical block, offset in block) through a block table.
fn resolve(table: &[u32], block_size: u32, token: u64) -> (u32, u32) {
    let bs = block_size as u64;
    (table[(token / bs) as usize], (token % bs) as u32)
}

fn check_side(name: &str, layout: &Layout, table: &[u32], start: u64, tokens: u64) -> Result<()> {
    let g = layout.geometry();
    if table.is_empty() {
        return invalid(format!("{name} block table is empty"));
    }
    if table.iter().any(|&b| b >= g.blocks) {
        return invalid(format!("{name} block id out of range"));
    }
    let capacity = table.len() as u64 * g.block_size as u64;
    if start.checked_add(tokens).is_none_or(|end| end > capacity) {
        return invalid(format!("{name} token range exceeds its block table"));
    }
    Ok(())
}

/// Build the byte-span plan that copies `sel` from a `src`-shaped buffer to a `dst`-shaped one.
pub fn plan(src: &KvSpec, dst: &KvSpec, sel: &Selection) -> Result<Plan> {
    src.compatible(dst)?;
    let (sl, dl) = (&src.layout, &dst.layout);
    let g = *sl.geometry();
    if sel.layers.is_empty() || sel.layers.end > g.layers {
        return invalid("layer range is empty or out of range");
    }
    let tokens = match sel.tokens {
        Some(n) => n,
        None => (sel.src_blocks.len() as u64 * g.block_size as u64).saturating_sub(sel.src_start),
    };
    if tokens == 0 {
        return invalid("empty token range");
    }
    check_side("source", sl, &sel.src_blocks, sel.src_start, tokens)?;
    check_side("destination", dl, &sel.dst_blocks, sel.dst_start, tokens)?;

    let row = g.row_bytes();
    // Walk (layer, plane, token, head) nested in the source's physical order (largest stride
    // outermost; region-selecting axes count as infinite), so consecutive rows are usually
    // adjacent on both sides and coalesce into few spans.
    let st = sl.strides();
    let region_stride =
        |axis_split: bool, s: u64, rank: u64| if axis_split { u64::MAX - rank } else { s };
    let (split_layer, split_kv) = match sl.split() {
        Split::None => (false, false),
        Split::Layer => (true, false),
        Split::LayerKv => (true, true),
    };
    let first_layer = sel.layers.start as u64;
    let mut axes = [
        (
            region_stride(split_layer, st.layer, 0),
            sel.layers.len() as u64,
            0usize,
        ),
        (region_stride(split_kv, st.kv, 1), g.kv.planes() as u64, 1),
        (st.token, tokens, 2),
        (st.head, g.heads as u64, 3),
    ];
    axes.sort_by_key(|a| std::cmp::Reverse(a.0));
    let extents = axes.map(|a| a.1);
    let total: u64 = extents.iter().product();
    let mut spans: Vec<Span> = Vec::new();
    for flat in 0..total {
        // Decode the flat index into the four (sorted) axis positions.
        let mut idx = [0u64; 4];
        let mut rest = flat;
        for k in (0..4).rev() {
            idx[axes[k].2] = rest % extents[k];
            rest /= extents[k];
        }
        let (layer, kv, t, h) = (
            (first_layer + idx[0]) as u32,
            idx[1] as u32,
            idx[2],
            idx[3] as u32,
        );
        let (sb, stok) = resolve(&sel.src_blocks, g.block_size, sel.src_start + t);
        let (db, dtok) = resolve(&sel.dst_blocks, dl.geometry().block_size, sel.dst_start + t);
        let (sr, so) = sl.locate(layer, kv, sb, stok, h);
        let (dr, dof) = dl.locate(layer, kv, db, dtok, h);
        match spans.last_mut() {
            Some(p)
                if p.src == sr
                    && p.dst == dr
                    && p.src_offset + p.len == so
                    && p.dst_offset + p.len == dof =>
            {
                p.len += row
            }
            _ => spans.push(Span {
                src: sr,
                src_offset: so,
                dst: dr,
                dst_offset: dof,
                len: row,
            }),
        }
    }
    let bytes = spans.iter().map(|s| s.len).sum();
    Ok(Plan {
        spans,
        bytes,
        layers: sel.layers.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Representation;
    use crate::kv::layout::{Axis, Geometry, KvKind};
    use std::collections::HashMap;

    fn spec(layout: Layout) -> KvSpec {
        KvSpec::new("m", "r", layout).unwrap()
    }

    fn g(kv: KvKind, blocks: u32, block_size: u32) -> Geometry {
        Geometry {
            representation: Representation::Fp16,
            kv,
            layers: 3,
            blocks,
            block_size,
            heads: 2,
            head_dim: 4,
        }
    }

    /// Execute a plan on byte maps and compare against a row-by-row reference copy.
    fn check(src: &KvSpec, dst: &KvSpec, sel: &Selection) -> Plan {
        let p = plan(src, dst, sel).unwrap();
        let (sl, dl) = (&src.layout, &dst.layout);
        let gm = *sl.geometry();
        let row = gm.row_bytes();
        let tokens = sel
            .tokens
            .unwrap_or(sel.src_blocks.len() as u64 * gm.block_size as u64 - sel.src_start);
        // Expected destination byte -> source byte.
        let mut expect: HashMap<(u32, u64), (u32, u64)> = HashMap::new();
        for l in sel.layers.clone() {
            for k in 0..gm.kv.planes() {
                for t in 0..tokens {
                    for h in 0..gm.heads {
                        let (sb, st) = resolve(&sel.src_blocks, gm.block_size, sel.src_start + t);
                        let (db, dt) =
                            resolve(&sel.dst_blocks, dl.geometry().block_size, sel.dst_start + t);
                        let (sr, so) = sl.locate(l, k, sb, st, h);
                        let (dr, d) = dl.locate(l, k, db, dt, h);
                        for x in 0..row {
                            expect.insert((dr, d + x), (sr, so + x));
                        }
                    }
                }
            }
        }
        let mut got: HashMap<(u32, u64), (u32, u64)> = HashMap::new();
        for s in p.spans() {
            assert!((s.src as usize) < sl.region_count() && (s.dst as usize) < dl.region_count());
            assert!(s.src_offset + s.len <= sl.region_bytes());
            assert!(s.dst_offset + s.len <= dl.region_bytes());
            for x in 0..s.len {
                let prev = got.insert((s.dst, s.dst_offset + x), (s.src, s.src_offset + x));
                assert!(prev.is_none(), "destination byte written twice");
            }
        }
        assert_eq!(got, expect, "plan coverage differs from the reference");
        assert_eq!(p.bytes(), expect.len() as u64);
        p
    }

    #[test]
    fn token_major_prefix_is_one_span_per_layer() {
        use Axis::*;
        let tm = |cap| {
            let geo = g(KvKind::KeyValue, 1, cap);
            spec(Layout::contiguous(geo, [Layer, Block, Token, Kv, Head], Split::Layer).unwrap())
        };
        let p = check(&tm(16), &tm(32), &Selection::prefix(0..3, 5));
        assert_eq!(p.spans().len(), 3);
        assert_eq!(p.bytes(), 3 * 5 * 2 * 2 * 8);
    }

    #[test]
    fn bhsd_prefix_spans_per_head() {
        let s = spec(Layout::bhsd(g(KvKind::KeyValue, 1, 16)).unwrap());
        let d = spec(Layout::bhsd(g(KvKind::KeyValue, 1, 64)).unwrap());
        let p = check(&s, &d, &Selection::prefix(0..3, 7));
        // 3 layers x 2 planes x 2 heads, each a contiguous run of 7 tokens.
        assert_eq!(p.spans().len(), 12);
        assert!(p.spans().iter().all(|s| s.len == 7 * 8));
    }

    #[test]
    fn paged_blocks_between_layouts() {
        let s = spec(Layout::layer_separate(g(KvKind::KeyValue, 8, 4)).unwrap());
        let d = spec(Layout::block_major(g(KvKind::KeyValue, 6, 4)).unwrap());
        check(
            &s,
            &d,
            &Selection::blocks(0..3, vec![5, 1, 7], vec![0, 4, 2]),
        );
        check(&s, &d, &Selection::blocks(1..2, vec![3], vec![5]));
        // Different block sizes: paged source gathered into a contiguous destination.
        let c = spec(Layout::layer_major(g(KvKind::KeyValue, 1, 32)).unwrap());
        let sel = Selection {
            layers: 0..3,
            src_blocks: vec![6, 2, 3],
            dst_blocks: vec![0],
            src_start: 2,
            dst_start: 9,
            tokens: Some(9),
        };
        check(&s, &c, &sel);
    }

    #[test]
    fn mla_latent_and_layer_streaming_spans() {
        let geo = Geometry {
            heads: 1,
            head_dim: 16,
            ..g(KvKind::Latent, 4, 8)
        };
        let s = spec(Layout::layer_separate(geo).unwrap());
        let d = spec(Layout::block_major(geo).unwrap());
        let p = check(&s, &d, &Selection::blocks(0..3, vec![0, 1], vec![3, 2]));
        assert_eq!(p.bytes(), 3 * 2 * 8 * 32);

        // pd_batching slab [kv, layer, block, head, token] -> per-layer tensors, partial span.
        use Axis::*;
        let slab_geo = g(KvKind::KeyValue, 1, 12);
        let slab = spec(
            Layout::contiguous(slab_geo, [Kv, Layer, Block, Head, Token], Split::None).unwrap(),
        );
        let dst = spec(Layout::bhsd(g(KvKind::KeyValue, 1, 20)).unwrap());
        check(&slab, &dst, &Selection::contiguous(1..3, 2, 7, 4));
    }

    #[test]
    fn rejections() {
        let s = spec(Layout::layer_separate(g(KvKind::KeyValue, 4, 4)).unwrap());
        let other_model = KvSpec::new("x", "r", s.layout.clone()).unwrap();
        assert!(plan(&s, &other_model, &Selection::blocks(0..1, vec![0], vec![0])).is_err());
        for sel in [
            Selection::blocks(0..0, vec![0], vec![0]),
            Selection::blocks(0..4, vec![0], vec![0]),
            Selection::blocks(0..1, vec![4], vec![0]),
            Selection::blocks(0..1, vec![0, 1], vec![0]),
            Selection::blocks(0..1, vec![], vec![0]),
            Selection::contiguous(0..1, 0, 0, 0),
            Selection::contiguous(0..1, 3, 0, 2),
        ] {
            assert!(plan(&s, &s, &sel).is_err(), "{sel:?}");
        }
    }
}
