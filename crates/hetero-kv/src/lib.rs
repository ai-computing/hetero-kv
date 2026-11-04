//! # hetero-kv
//!
//! Layered KV-cache transport between heterogeneous accelerators. The crate only moves KV
//! bytes; model execution, batching and serving stay in the engines that use it.
//!
//! | Layer | Module | Owns |
//! |---|---|---|
//! | L1 | [`memory`] | regions, ownership state machine, registration and region descriptors |
//! | L2 | [`engine`] | `CopyEngine`, capability matrix, explicit routes, transfer handles, host engine |
//! | L3 | [`kv`] | `KvSpec`, layouts, byte-span plans, layer progress, KV buffers/descriptors, events |
//! | L4 | [`session`] | reserve -> transfer -> fence -> admit -> finish, cancel, poison; agents |
//!
//! Dependencies point downwards only (L4 -> L3 -> L2 -> L1). [`wire`] is the shared versioned
//! encoding. Bindings (C ABI, Python) live in sibling crates.
//!
//! ```
//! use std::sync::Arc;
//! use hetero_kv::engine::{Direction, HostEngine, Route};
//! use hetero_kv::kv::{Geometry, KvBuffer, KvKind, KvSpec, Layout, Representation, Selection};
//! use hetero_kv::memory::{MemoryKind, Region};
//! use hetero_kv::session::Session;
//!
//! let g = Geometry { representation: Representation::Bf16, kv: KvKind::KeyValue,
//!                    layers: 2, blocks: 1, block_size: 16, heads: 2, head_dim: 8 };
//! let spec = KvSpec::new("model", "rev", Layout::bhsd(g)?)?;
//! let alloc = |s: &KvSpec| -> hetero_kv::Result<KvBuffer> {
//!     let n = s.layout.region_bytes() as usize;
//!     let regions = (0..s.layout.region_count()).map(|_| Region::host(n)).collect::<Result<_, _>>()?;
//!     KvBuffer::new(s.clone(), regions)
//! };
//! let (src, dst) = (alloc(&spec)?, alloc(&spec)?);
//!
//! let route = Route::select(Arc::new(HostEngine::new()), MemoryKind::Host, MemoryKind::Host,
//!                           Representation::Bf16)?;
//! let session = Session::new(route);
//! session.reserve("req-1", vec![dst.share()])?;
//! session.transfer("req-1", 0, &src.share(), &[Selection::prefix(0..2, 5)], Direction::Write)?;
//! let receipt = session.fence("req-1")?;
//! session.admit("req-1", || Ok(()))?;   // make the cache visible to the consumer
//! session.finish("req-1", true)?;
//! assert_eq!(receipt.bytes, 2 * 2 * 2 * 5 * 16);
//! # Ok::<(), hetero_kv::Error>(())
//! ```

#![warn(missing_docs)]

pub mod engine;
mod error;
pub mod kv;
pub mod memory;
pub mod session;
pub mod wire;

pub use error::{Error, Result};
