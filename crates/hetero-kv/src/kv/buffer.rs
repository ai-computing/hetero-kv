//! KV buffers (a spec plus its regions) and their serializable descriptors.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::layout::KvSpec;
use crate::error::{invalid, Error, Result};
use crate::memory::{Lent, MemoryKind, Quiesced, Region, RegionDesc, RegionRef, Registry};

/// Content hash of a block (e.g. vLLM's prefix-chain block hash): an integer or raw bytes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BlockHash {
    /// Integer hash (vLLM `int` block hashes, llm-d's uint64 keys).
    Int(u64),
    /// Byte-string hash (e.g. sha256-based block hashes).
    Bytes(Vec<u8>),
}

/// A failed [`KvBuffer::reclaim`]: the unchanged buffer and the reason.
#[derive(Debug)]
pub struct BufferReclaim {
    /// The buffer, still holding every region.
    pub buffer: KvBuffer,
    /// Why it cannot be reclaimed yet.
    pub error: Error,
}

/// A KV buffer: its spec and the regions the layout addresses, all of one memory kind.
/// Owns the [`Lent`] handles; give it back with [`KvBuffer::reclaim`].
pub struct KvBuffer {
    spec: KvSpec,
    regions: Vec<Lent>,
}

/// Shared reference to a buffer (local, or resolved from a descriptor). Holding one keeps the
/// owner from reclaiming the regions.
#[derive(Clone, Debug)]
pub struct KvBufferRef {
    spec: KvSpec,
    regions: Vec<RegionRef>,
}

fn check_regions<'a>(
    spec: &KvSpec,
    lens_kinds: impl ExactSizeIterator<Item = (usize, MemoryKind)> + 'a,
) -> Result<()> {
    let layout = &spec.layout;
    if lens_kinds.len() != layout.region_count() {
        return invalid(format!(
            "layout needs {} regions, got {}",
            layout.region_count(),
            lens_kinds.len()
        ));
    }
    let mut kind = None;
    for (len, k) in lens_kinds {
        if *kind.get_or_insert(k) != k {
            return invalid("all regions of a KV buffer must be one memory kind");
        }
        if (len as u64) < layout.region_bytes() {
            return invalid(format!(
                "region of {len} bytes is smaller than the layout's {}",
                layout.region_bytes()
            ));
        }
    }
    Ok(())
}

impl KvBuffer {
    /// Check that `regions` fit `spec` without consuming them.
    pub fn validate(spec: &KvSpec, regions: &[Region]) -> Result<()> {
        check_regions(spec, regions.iter().map(|r| (r.len(), r.kind())))
    }

    /// Build a buffer, lending the regions. Regions are dropped if validation fails; call
    /// [`KvBuffer::validate`] first to keep them.
    pub fn new(spec: KvSpec, regions: Vec<Region>) -> Result<KvBuffer> {
        Self::validate(&spec, &regions)?;
        Ok(KvBuffer {
            spec,
            regions: regions.into_iter().map(Region::lend).collect(),
        })
    }

    /// Spec.
    pub fn spec(&self) -> &KvSpec {
        &self.spec
    }

    /// Memory kind of the regions.
    pub fn kind(&self) -> MemoryKind {
        self.regions[0].kind()
    }

    /// Lent region handles (for phase inspection).
    pub fn regions(&self) -> &[Lent] {
        &self.regions
    }

    /// Shared reference for sessions and registration.
    pub fn share(&self) -> KvBufferRef {
        KvBufferRef {
            spec: self.spec.clone(),
            regions: self.regions.iter().map(Lent::share).collect(),
        }
    }

    /// Get every region back (all or nothing).
    pub fn reclaim(self) -> std::result::Result<Vec<Region>, Box<BufferReclaim>> {
        self.reclaim_with(None)
    }

    /// Like [`KvBuffer::reclaim`], also accepting regions left uncertain by a quiesced engine.
    pub fn reclaim_quiesced(
        self,
        proof: &Quiesced,
    ) -> std::result::Result<Vec<Region>, Box<BufferReclaim>> {
        self.reclaim_with(Some(proof))
    }

    fn reclaim_with(
        self,
        proof: Option<&Quiesced>,
    ) -> std::result::Result<Vec<Region>, Box<BufferReclaim>> {
        let KvBuffer { spec, regions } = self;
        let mut done: Vec<Region> = Vec::new();
        let mut rest = regions.into_iter();
        for lent in rest.by_ref() {
            let r = match proof {
                Some(p) => lent.reclaim_quiesced(p),
                None => lent.reclaim(),
            };
            match r {
                Ok(region) => done.push(region),
                Err(fail) => {
                    // Re-lend what was already reclaimed so the buffer is unchanged.
                    let mut regions: Vec<Lent> = done.into_iter().map(Region::lend).collect();
                    regions.push(fail.lent);
                    regions.extend(rest);
                    return Err(Box::new(BufferReclaim {
                        buffer: KvBuffer { spec, regions },
                        error: fail.error,
                    }));
                }
            }
        }
        Ok(done)
    }
}

impl std::fmt::Debug for KvBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KvBuffer({}, {} regions)",
            self.kind(),
            self.regions.len()
        )
    }
}

impl KvBufferRef {
    /// Spec.
    pub fn spec(&self) -> &KvSpec {
        &self.spec
    }

    /// Memory kind of the regions.
    pub fn kind(&self) -> MemoryKind {
        self.regions[0].kind()
    }

    /// Region references in layout order.
    pub fn regions(&self) -> &[RegionRef] {
        &self.regions
    }

    /// Register the regions with `registry` and describe the buffer for a peer.
    /// `block_hashes` optionally maps block ids to content hashes.
    pub fn export(
        &self,
        registry: &Registry,
        block_hashes: BTreeMap<u32, BlockHash>,
    ) -> Result<KvDescriptor> {
        let blocks = self.spec.layout.geometry().blocks;
        if block_hashes.keys().any(|&b| b >= blocks) {
            return invalid("block hash names a block outside the layout");
        }
        Ok(KvDescriptor {
            spec: self.spec.clone(),
            regions: self.regions.iter().map(|r| registry.register(r)).collect(),
            block_hashes,
        })
    }
}

/// Serializable description of a registered KV buffer: what a peer (another engine, a KV
/// manager, a P/D sidecar) needs to address it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvDescriptor {
    /// Spec of the buffer.
    pub spec: KvSpec,
    /// Registered regions in layout order.
    pub regions: Vec<RegionDesc>,
    /// Optional content hashes by block id.
    #[serde(default)]
    pub block_hashes: BTreeMap<u32, BlockHash>,
}

impl KvDescriptor {
    /// Versioned wire encoding.
    pub fn to_bytes(&self) -> Vec<u8> {
        crate::wire::encode("kv_buffer", self)
    }

    /// Decode [`KvDescriptor::to_bytes`] and check it against its layout.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let d: KvDescriptor = crate::wire::decode("kv_buffer", bytes)?;
        check_regions(&d.spec, d.regions.iter().map(|r| (r.len as usize, r.kind)))?;
        Ok(d)
    }

    /// Resolve to the registered regions (in-process registries only in P0).
    pub fn resolve(&self) -> Result<KvBufferRef> {
        let regions = self
            .regions
            .iter()
            .map(Registry::resolve)
            .collect::<Result<Vec<_>>>()?;
        Ok(KvBufferRef {
            spec: self.spec.clone(),
            regions,
        })
    }

    /// Drop the registrations made by [`KvBufferRef::export`].
    pub fn deregister(&self, registry: &Registry) -> Result<()> {
        self.regions.iter().try_for_each(|r| registry.deregister(r))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Representation;
    use crate::kv::layout::{Geometry, KvKind, Layout};

    pub(crate) fn spec() -> KvSpec {
        let g = Geometry {
            representation: Representation::Bf16,
            kv: KvKind::KeyValue,
            layers: 2,
            blocks: 4,
            block_size: 4,
            heads: 1,
            head_dim: 8,
        };
        KvSpec::new("m", "r", Layout::layer_separate(g).unwrap()).unwrap()
    }

    fn regions(n: usize, len: usize) -> Vec<Region> {
        (0..n).map(|_| Region::host(len).unwrap()).collect()
    }

    #[test]
    fn buffer_validation_and_reclaim() {
        let s = spec();
        let need = s.layout.region_bytes() as usize;
        assert!(KvBuffer::new(s.clone(), regions(1, need)).is_err());
        assert!(KvBuffer::new(s.clone(), regions(2, need - 1)).is_err());
        let buf = KvBuffer::new(s, regions(2, need)).unwrap();
        let r = buf.share();
        let BufferReclaim {
            buffer: buf,
            error: e,
        } = *buf.reclaim().unwrap_err();
        assert!(matches!(e, Error::Busy(_)));
        assert_eq!(buf.regions().len(), 2);
        drop(r);
        assert_eq!(buf.reclaim().unwrap().len(), 2);
    }

    #[test]
    fn descriptor_round_trip_with_hashes() {
        let s = spec();
        let need = s.layout.region_bytes() as usize;
        let buf = KvBuffer::new(s, regions(2, need)).unwrap();
        let reg = Registry::new();
        let hashes = BTreeMap::from([(0, BlockHash::Int(42)), (3, BlockHash::Bytes(vec![1, 2]))]);
        let desc = buf.share().export(&reg, hashes.clone()).unwrap();
        let back = KvDescriptor::from_bytes(&desc.to_bytes()).unwrap();
        assert_eq!(back, desc);
        assert_eq!(back.block_hashes, hashes);
        let r = back.resolve().unwrap();
        assert_eq!(r.regions().len(), 2);
        drop(r);
        assert!(buf
            .share()
            .export(&reg, BTreeMap::from([(9, BlockHash::Int(1))]))
            .is_err());
        desc.deregister(&reg).unwrap();
        assert!(back.resolve().is_err());
        buf.reclaim().unwrap();
    }
}
