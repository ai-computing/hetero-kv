//! Physical KV layouts and the KV specification.
//!
//! A layout places every row — the `head_dim` elements of one (layer, plane, block, token, head)
//! — at a byte offset inside one of the buffer's regions. Planes are K and V for ordinary
//! attention ([`KvKind::KeyValue`]) or the single latent of MLA ([`KvKind::Latent`]).
//!
//! * **Regions.** [`Split`] says which leading axes select a region: one region for the whole
//!   cache, one per layer (vLLM per-layer tensors), or one per layer and plane (separate K and
//!   V tensors, as in Optimus `KVCacheManager`). "Fused" K/V means the planes share a region
//!   and differ by the `kv` stride.
//! * **Strides.** Byte strides of the remaining axes inside a region. [`Layout::contiguous`]
//!   derives them from an axis order; [`Layout::strided`] accepts explicit (e.g. padded) ones.
//! * **Blocks.** Paged caches have `blocks` blocks of `block_size` tokens. A contiguous cache is
//!   one block whose size is the token capacity, so a token-major prefix copy is the special case
//!   "block 0, tokens 0..n".

use serde::{Deserialize, Serialize};

use crate::engine::Representation;
use crate::error::{invalid, Error, Result};

/// Version of the layout descriptor semantics. Bump if offsets would be computed differently.
pub const LAYOUT_VERSION: u32 = 1;

/// Logical axes of a KV cache; the element (`head_dim`) axis is always innermost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Axis {
    /// Transformer layer.
    Layer,
    /// K/V plane (or the single MLA latent plane).
    Kv,
    /// Cache block (page).
    Block,
    /// Token within a block.
    Token,
    /// KV head.
    Head,
}

impl Axis {
    const ALL: [Axis; 5] = [Axis::Layer, Axis::Kv, Axis::Block, Axis::Token, Axis::Head];

    /// Parse a lowercase axis name.
    pub fn parse(s: &str) -> Result<Axis> {
        Axis::ALL
            .into_iter()
            .find(|a| a.as_str() == s)
            .ok_or_else(|| Error::Invalid(format!("unknown axis {s:?}")))
    }

    /// Lowercase name.
    pub fn as_str(self) -> &'static str {
        match self {
            Axis::Layer => "layer",
            Axis::Kv => "kv",
            Axis::Block => "block",
            Axis::Token => "token",
            Axis::Head => "head",
        }
    }
}

/// Planes per layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvKind {
    /// Separate K and V planes.
    KeyValue,
    /// One latent plane (MLA); `heads` is usually 1 and `head_dim` the latent width.
    Latent,
}

impl KvKind {
    /// Number of planes.
    pub fn planes(self) -> u32 {
        match self {
            KvKind::KeyValue => 2,
            KvKind::Latent => 1,
        }
    }
}

/// Which leading axes select a region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    /// One region holds the whole cache.
    None,
    /// One region per layer.
    Layer,
    /// One region per (layer, plane).
    LayerKv,
}

/// Logical geometry shared by every layout of the same model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Geometry {
    /// Element encoding.
    pub representation: Representation,
    /// K/V planes or MLA latent.
    pub kv: KvKind,
    /// Layers in this buffer.
    pub layers: u32,
    /// Blocks (pages); 1 for a contiguous cache.
    pub blocks: u32,
    /// Tokens per block; the token capacity for a contiguous cache.
    pub block_size: u32,
    /// KV heads (after any tensor-parallel split).
    pub heads: u32,
    /// Elements per head row.
    pub head_dim: u32,
}

impl Geometry {
    fn extent(&self, axis: Axis) -> u32 {
        match axis {
            Axis::Layer => self.layers,
            Axis::Kv => self.kv.planes(),
            Axis::Block => self.blocks,
            Axis::Token => self.block_size,
            Axis::Head => self.heads,
        }
    }

    /// Bytes of one head row.
    pub fn row_bytes(&self) -> u64 {
        self.head_dim as u64 * self.representation.element_bytes()
    }
}

/// Byte strides inside a region. Strides of axes consumed by [`Split`] must be 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Strides {
    /// Per layer.
    pub layer: u64,
    /// Per plane.
    pub kv: u64,
    /// Per block.
    pub block: u64,
    /// Per token.
    pub token: u64,
    /// Per head.
    pub head: u64,
}

impl Strides {
    fn get(&self, axis: Axis) -> u64 {
        match axis {
            Axis::Layer => self.layer,
            Axis::Kv => self.kv,
            Axis::Block => self.block,
            Axis::Token => self.token,
            Axis::Head => self.head,
        }
    }
    fn set(&mut self, axis: Axis, v: u64) {
        match axis {
            Axis::Layer => self.layer = v,
            Axis::Kv => self.kv = v,
            Axis::Block => self.block = v,
            Axis::Token => self.token = v,
            Axis::Head => self.head = v,
        }
    }
}

/// A validated physical layout.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "LayoutWire")]
pub struct Layout {
    version: u32,
    geometry: Geometry,
    split: Split,
    strides: Strides,
    region_bytes: u64,
}

#[derive(Deserialize)]
struct LayoutWire {
    version: u32,
    geometry: Geometry,
    split: Split,
    strides: Strides,
    region_bytes: u64,
}

impl TryFrom<LayoutWire> for Layout {
    type Error = Error;
    fn try_from(w: LayoutWire) -> Result<Layout> {
        if w.version != LAYOUT_VERSION {
            return Err(Error::Unsupported(format!(
                "layout version {} (this build reads {LAYOUT_VERSION})",
                w.version
            )));
        }
        Layout::strided(w.geometry, w.split, w.strides, w.region_bytes)
    }
}

fn split_axes(split: Split) -> &'static [Axis] {
    match split {
        Split::None => &[],
        Split::Layer => &[Axis::Layer],
        Split::LayerKv => &[Axis::Layer, Axis::Kv],
    }
}

impl Layout {
    /// Dense layout from an outer-to-inner axis order (all five axes). Split axes must lead.
    pub fn contiguous(geometry: Geometry, order: [Axis; 5], split: Split) -> Result<Layout> {
        if Axis::ALL.iter().any(|a| !order.contains(a)) {
            return invalid("axis order must name each of layer, kv, block, token, head once");
        }
        let lead = split_axes(split);
        if order[..lead.len()] != *lead {
            return invalid("split axes must lead the axis order");
        }
        let mut strides = Strides::default();
        let mut running = geometry.row_bytes();
        for &axis in order[lead.len()..].iter().rev() {
            strides.set(axis, running);
            running = running
                .checked_mul(geometry.extent(axis) as u64)
                .ok_or_else(|| Error::Invalid("layout too large".into()))?;
        }
        Layout::strided(geometry, split, strides, running)
    }

    /// Layout with explicit byte strides and region size.
    pub fn strided(
        geometry: Geometry,
        split: Split,
        strides: Strides,
        region_bytes: u64,
    ) -> Result<Layout> {
        let g = geometry;
        if [g.layers, g.blocks, g.block_size, g.heads, g.head_dim].contains(&0) {
            return invalid("positive KV dimensions required");
        }
        let lead = split_axes(split);
        let mut inner: Vec<(u64, u64)> = Vec::new(); // (stride, extent) of in-region axes
        for axis in Axis::ALL {
            let (stride, extent) = (strides.get(axis), g.extent(axis) as u64);
            if lead.contains(&axis) {
                if stride != 0 {
                    return invalid(format!("stride of split axis {} must be 0", axis.as_str()));
                }
            } else if extent > 1 {
                inner.push((stride, extent));
            }
        }
        // Rows must not overlap: each axis must step over everything nested inside it.
        inner.sort_unstable();
        let mut reach = g.row_bytes();
        for &(stride, extent) in &inner {
            if stride < reach {
                return invalid("strides overlap rows");
            }
            reach = stride
                .checked_mul(extent - 1)
                .and_then(|x| x.checked_add(reach))
                .ok_or_else(|| Error::Invalid("layout too large".into()))?;
        }
        if reach > region_bytes {
            return invalid("strides exceed the region size");
        }
        Ok(Layout {
            version: LAYOUT_VERSION,
            geometry,
            split,
            strides,
            region_bytes,
        })
    }

    /// One region per layer, `[kv, block, token, head, dim]` inside: vLLM FlashAttention-style
    /// per-layer tensors (`[2, num_blocks, block_size, heads, head_dim]`), Dynamo
    /// `LayerSeparate`.
    pub fn layer_separate(g: Geometry) -> Result<Layout> {
        use Axis::*;
        Self::contiguous(g, [Layer, Kv, Block, Token, Head], Split::Layer)
    }

    /// One region, `[block, layer, kv, token, head, dim]`: each block holds every layer
    /// (LMCache chunks, Dynamo `FullyContiguous`). Best for block-granular offload.
    pub fn block_major(g: Geometry) -> Result<Layout> {
        use Axis::*;
        Self::contiguous(g, [Block, Layer, Kv, Token, Head], Split::None)
    }

    /// One region, `[layer, kv, block, token, head, dim]`.
    pub fn layer_major(g: Geometry) -> Result<Layout> {
        use Axis::*;
        Self::contiguous(g, [Layer, Kv, Block, Token, Head], Split::None)
    }

    /// One region per (layer, plane), `[block, head, token, dim]` inside: Hugging Face /
    /// Optimus `KVCacheManager` tensors `[batch, heads, seq, head_dim]` with `blocks = batch`.
    pub fn bhsd(g: Geometry) -> Result<Layout> {
        use Axis::*;
        Self::contiguous(g, [Layer, Kv, Block, Head, Token], Split::LayerKv)
    }

    /// Layout descriptor version.
    pub fn version(&self) -> u32 {
        self.version
    }
    /// Logical geometry.
    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }
    /// Region split.
    pub fn split(&self) -> Split {
        self.split
    }
    /// In-region strides.
    pub fn strides(&self) -> &Strides {
        &self.strides
    }
    /// Minimum bytes per region.
    pub fn region_bytes(&self) -> u64 {
        self.region_bytes
    }

    /// Number of regions a buffer with this layout has.
    pub fn region_count(&self) -> usize {
        let g = &self.geometry;
        match self.split {
            Split::None => 1,
            Split::Layer => g.layers as usize,
            Split::LayerKv => (g.layers * g.kv.planes()) as usize,
        }
    }

    /// (region index, byte offset) of a row. Callers keep indices in range.
    pub(crate) fn locate(
        &self,
        layer: u32,
        kv: u32,
        block: u32,
        token: u32,
        head: u32,
    ) -> (u32, u64) {
        let s = &self.strides;
        let region = match self.split {
            Split::None => 0,
            Split::Layer => layer,
            Split::LayerKv => layer * self.geometry.kv.planes() + kv,
        };
        let offset = layer as u64 * s.layer
            + kv as u64 * s.kv
            + block as u64 * s.block
            + token as u64 * s.token
            + head as u64 * s.head;
        (region, offset)
    }
}

/// What a KV buffer holds: model identity plus physical layout. Two buffers can exchange
/// data only if [`KvSpec::compatible`] holds; block counts, block sizes and strides may differ.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KvSpec {
    /// Model identifier.
    pub model_id: String,
    /// Immutable model revision.
    pub revision: String,
    /// Physical layout (includes representation and layout version).
    pub layout: Layout,
}

impl KvSpec {
    /// Validated spec; model id and revision must be non-blank.
    pub fn new(
        model_id: impl Into<String>,
        revision: impl Into<String>,
        layout: Layout,
    ) -> Result<KvSpec> {
        let (model_id, revision) = (model_id.into(), revision.into());
        if model_id.trim().is_empty() || revision.trim().is_empty() {
            return invalid("explicit model id and revision required");
        }
        Ok(KvSpec {
            model_id,
            revision,
            layout,
        })
    }

    /// Element encoding.
    pub fn representation(&self) -> Representation {
        self.layout.geometry.representation
    }

    /// Layer count.
    pub fn layers(&self) -> u32 {
        self.layout.geometry.layers
    }

    /// Ok if rows of `self` and `other` mean the same thing (model, revision, representation,
    /// layout version, planes, layers, heads, head_dim).
    pub fn compatible(&self, other: &KvSpec) -> Result<()> {
        let (a, b) = (&self.layout.geometry, &other.layout.geometry);
        let what = if self.model_id != other.model_id || self.revision != other.revision {
            "model revision"
        } else if a.representation != b.representation {
            "representation"
        } else if self.layout.version != other.layout.version {
            "layout version"
        } else if (a.kv, a.layers, a.heads, a.head_dim) != (b.kv, b.layers, b.heads, b.head_dim) {
            "KV geometry"
        } else {
            return Ok(());
        };
        invalid(format!("source and destination KV specs differ in {what}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn geom(blocks: u32, block_size: u32) -> Geometry {
        Geometry {
            representation: Representation::Bf16,
            kv: KvKind::KeyValue,
            layers: 3,
            blocks,
            block_size,
            heads: 2,
            head_dim: 4,
        }
    }

    #[test]
    fn contiguous_strides() {
        let l = Layout::layer_separate(geom(5, 16)).unwrap();
        assert_eq!(l.region_count(), 3);
        // [kv, block, token, head, dim] with 2-byte elements.
        assert_eq!(l.strides().head, 8);
        assert_eq!(l.strides().token, 16);
        assert_eq!(l.strides().block, 16 * 16);
        assert_eq!(l.strides().kv, 5 * 16 * 16);
        assert_eq!(l.region_bytes(), 2 * 5 * 16 * 16);
        assert_eq!(l.locate(2, 1, 0, 0, 1), (2, 5 * 256 + 8));

        let b = Layout::bhsd(geom(1, 10)).unwrap();
        assert_eq!(b.region_count(), 6);
        assert_eq!((b.strides().token, b.strides().head), (8, 80));
    }

    #[test]
    fn rejects_bad_layouts() {
        use Axis::*;
        let g = geom(2, 4);
        assert!(Layout::contiguous(g, [Layer, Layer, Block, Token, Head], Split::None).is_err());
        assert!(Layout::contiguous(g, [Kv, Layer, Block, Token, Head], Split::Layer).is_err());
        assert!(Layout::layer_major(Geometry { heads: 0, ..g }).is_err());
        let s = *Layout::layer_major(g).unwrap().strides();
        // Overlapping strides.
        assert!(Layout::strided(g, Split::None, Strides { head: 4, ..s }, 1 << 20).is_err());
        // Region too small.
        assert!(Layout::strided(g, Split::None, s, 100).is_err());
        // Split axis with a stride.
        assert!(Layout::strided(g, Split::Layer, s, 1 << 20).is_err());
        // Padding is fine.
        let padded = Strides {
            token: s.token * 2,
            block: s.block * 2,
            kv: s.kv * 2,
            layer: s.layer * 2,
            ..s
        };
        assert!(Layout::strided(g, Split::None, padded, 1 << 20).is_ok());
    }

    #[test]
    fn layout_serde_validates() {
        let l = Layout::block_major(geom(4, 8)).unwrap();
        let json = serde_json::to_string(&l).unwrap();
        assert_eq!(serde_json::from_str::<Layout>(&json).unwrap(), l);
        let bad = json.replace("\"region_bytes\":", "\"region_bytes\":1,\"x\":");
        let broken: std::result::Result<Layout, _> = serde_json::from_str(&bad);
        assert!(broken.is_err());
    }

    #[test]
    fn spec_compatibility() {
        let a = KvSpec::new("m", "r", Layout::layer_separate(geom(8, 16)).unwrap()).unwrap();
        let b = KvSpec::new("m", "r", Layout::bhsd(geom(1, 128)).unwrap()).unwrap();
        a.compatible(&b).unwrap();
        let c = KvSpec::new("m", "other", b.layout.clone()).unwrap();
        assert!(a.compatible(&c).is_err());
        let fp16 = Geometry {
            representation: Representation::Fp16,
            ..geom(1, 128)
        };
        let d = KvSpec::new("m", "r", Layout::bhsd(fp16).unwrap()).unwrap();
        assert!(a.compatible(&d).is_err());
        assert!(KvSpec::new(" ", "r", b.layout).is_err());
    }
}
