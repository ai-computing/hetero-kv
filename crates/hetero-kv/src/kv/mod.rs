//! L3 KV: specs, layouts, layout -> byte-span plans, layer progress, buffers and descriptors,
//! and block events.

mod buffer;
mod events;
mod layout;
mod plan;
mod progress;

pub use buffer::{BlockHash, BufferReclaim, KvBuffer, KvBufferRef, KvDescriptor};
pub use events::{EventLog, KvEvent, KvEventBatch};
pub use layout::{Axis, Geometry, KvKind, KvSpec, Layout, Split, Strides, LAYOUT_VERSION};
pub use plan::{plan, Plan, Selection};
pub use progress::{LayerProgress, ProgressSpan};

pub use crate::engine::Representation;
