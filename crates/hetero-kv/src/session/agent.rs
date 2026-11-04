//! Agents: the identity a process (or engine instance) presents to peers.
//!
//! An agent owns a registry and an event log. Its [`AgentMeta`] is what a peer receives during a
//! handshake (e.g. through an llm-d P/D sidecar or vLLM `kv_transfer_params`): the agent id that
//! descriptors refer to and an opaque endpoint string. P0 resolves descriptors only between
//! agents in the same process; cross-process transport (unix-socket fd passing, NIXL) is P3.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::kv::{BlockHash, EventLog, KvBufferRef, KvDescriptor};
use crate::memory::Registry;
use crate::wire::WIRE_VERSION;

/// Serializable agent metadata exchanged at handshake.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentMeta {
    /// Agent id; equals the registry id in this agent's descriptors.
    pub agent_id: u64,
    /// Human-readable name (e.g. the engine id).
    pub name: String,
    /// Opaque endpoint for a later transport (e.g. "host:port"); unused in P0.
    pub endpoint: Option<String>,
    /// Wire version the agent speaks.
    pub wire_version: u32,
}

impl AgentMeta {
    /// Versioned wire encoding.
    pub fn to_bytes(&self) -> Vec<u8> {
        crate::wire::encode("agent", self)
    }

    /// Decode [`AgentMeta::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        crate::wire::decode("agent", bytes)
    }
}

/// A named memory domain with registration and KV events.
pub struct Agent {
    name: String,
    endpoint: Option<String>,
    registry: Registry,
    events: EventLog,
}

impl Agent {
    /// New agent with a fresh registry.
    pub fn new(name: impl Into<String>, endpoint: Option<String>) -> Agent {
        Agent {
            name: name.into(),
            endpoint,
            registry: Registry::new(),
            events: EventLog::new(),
        }
    }

    /// Agent id.
    pub fn id(&self) -> u64 {
        self.registry.id()
    }

    /// Name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Handshake metadata.
    pub fn meta(&self) -> AgentMeta {
        AgentMeta {
            agent_id: self.id(),
            name: self.name.clone(),
            endpoint: self.endpoint.clone(),
            wire_version: WIRE_VERSION,
        }
    }

    /// The agent's registry.
    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    /// The agent's event log.
    pub fn events(&self) -> &EventLog {
        &self.events
    }

    /// Register a buffer and describe it for peers.
    pub fn export(
        &self,
        buffer: &KvBufferRef,
        block_hashes: BTreeMap<u32, BlockHash>,
    ) -> Result<KvDescriptor> {
        buffer.export(&self.registry, block_hashes)
    }

    /// Drop the registrations of an exported descriptor.
    pub fn release(&self, desc: &KvDescriptor) -> Result<()> {
        desc.deregister(&self.registry)
    }

    /// Resolve a peer's descriptor (P0: peers in this process only).
    pub fn resolve(&self, desc: &KvDescriptor) -> Result<KvBufferRef> {
        desc.resolve()
    }
}
