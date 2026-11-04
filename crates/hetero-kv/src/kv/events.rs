//! KV block events for KV managers and KV-cache-aware routers.
//!
//! hetero-kv does not route or index blocks; it gives KV managers one event type and a log
//! they can publish into, and renders batches in the shape vLLM's `KVEventBatch` serializes to
//! (msgspec `array_like` + `tag`), which llm-d's KV-cache-aware router consumes. Encoding the
//! rendered value as msgpack and publishing it (ZMQ) is the consumer's job.

use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::buffer::BlockHash;

/// A change in where content-addressed blocks live.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum KvEvent {
    /// Blocks became available in `medium`.
    BlockStored {
        /// Hashes of the stored blocks, in chain order.
        block_hashes: Vec<BlockHash>,
        /// Hash of the block preceding the first stored block, if any.
        parent_block_hash: Option<BlockHash>,
        /// Tokens covered by the blocks.
        token_ids: Vec<u32>,
        /// Tokens per block.
        block_size: u32,
        /// LoRA adapter id, if any.
        lora_id: Option<u32>,
        /// Tier name (`MemoryKind::medium`, e.g. "GPU", "CPU").
        medium: Option<String>,
    },
    /// Blocks were evicted from `medium`.
    BlockRemoved {
        /// Hashes of the removed blocks.
        block_hashes: Vec<BlockHash>,
        /// Tier name.
        medium: Option<String>,
    },
    /// Blocks moved between tiers (offload / onload). Rendered for vLLM as removed + stored.
    BlockMoved {
        /// Hashes of the moved blocks.
        block_hashes: Vec<BlockHash>,
        /// Parent hash, as for `BlockStored`.
        parent_block_hash: Option<BlockHash>,
        /// Tokens covered by the blocks.
        token_ids: Vec<u32>,
        /// Tokens per block.
        block_size: u32,
        /// LoRA adapter id, if any.
        lora_id: Option<u32>,
        /// Tier the blocks left.
        from_medium: String,
        /// Tier the blocks now live in.
        to_medium: String,
    },
    /// Every block was dropped.
    AllBlocksCleared,
}

impl KvEvent {
    /// vLLM-shaped values (`[tag, field, ...]`) for this event; a move yields two.
    pub fn to_vllm(&self) -> Vec<Value> {
        match self {
            KvEvent::BlockStored {
                block_hashes,
                parent_block_hash,
                token_ids,
                block_size,
                lora_id,
                medium,
            } => vec![json!([
                "BlockStored",
                block_hashes,
                parent_block_hash,
                token_ids,
                block_size,
                lora_id,
                medium
            ])],
            KvEvent::BlockRemoved {
                block_hashes,
                medium,
            } => vec![json!(["BlockRemoved", block_hashes, medium])],
            KvEvent::BlockMoved {
                block_hashes,
                parent_block_hash,
                token_ids,
                block_size,
                lora_id,
                from_medium,
                to_medium,
            } => vec![
                json!(["BlockRemoved", block_hashes, from_medium]),
                json!([
                    "BlockStored",
                    block_hashes,
                    parent_block_hash,
                    token_ids,
                    block_size,
                    lora_id,
                    to_medium
                ]),
            ],
            KvEvent::AllBlocksCleared => vec![json!(["AllBlocksCleared"])],
        }
    }
}

/// A batch of events with a timestamp, as published to a router.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvEventBatch {
    /// Seconds since the Unix epoch.
    pub ts: f64,
    /// Events in publication order.
    pub events: Vec<KvEvent>,
    /// Data-parallel rank of the publisher, if any.
    pub data_parallel_rank: Option<i32>,
}

impl KvEventBatch {
    /// `[ts, [event, ...], data_parallel_rank]`, the array form of vLLM's `KVEventBatch`.
    /// Field order follows vLLM v0.10/v0.11 `vllm/distributed/kv_events.py`; check it against the
    /// vLLM version you pin.
    pub fn to_vllm(&self) -> Value {
        let events: Vec<Value> = self.events.iter().flat_map(KvEvent::to_vllm).collect();
        json!([self.ts, events, self.data_parallel_rank])
    }
}

/// Thread-safe append-only event queue.
#[derive(Default)]
pub struct EventLog {
    events: Mutex<Vec<KvEvent>>,
}

impl EventLog {
    /// Empty log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an event.
    pub fn publish(&self, event: KvEvent) {
        self.events
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(event);
    }

    /// Take all events as a timestamped batch (`None` if empty).
    pub fn drain(&self, data_parallel_rank: Option<i32>) -> Option<KvEventBatch> {
        let events = std::mem::take(&mut *self.events.lock().unwrap_or_else(|p| p.into_inner()));
        if events.is_empty() {
            return None;
        }
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        Some(KvEventBatch {
            ts,
            events,
            data_parallel_rank,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vllm_shape() {
        let log = EventLog::new();
        assert!(log.drain(None).is_none());
        log.publish(KvEvent::BlockStored {
            block_hashes: vec![BlockHash::Int(1), BlockHash::Int(2)],
            parent_block_hash: None,
            token_ids: vec![5, 6, 7, 8],
            block_size: 2,
            lora_id: None,
            medium: Some("GPU".into()),
        });
        log.publish(KvEvent::BlockMoved {
            block_hashes: vec![BlockHash::Int(1)],
            parent_block_hash: None,
            token_ids: vec![5, 6],
            block_size: 2,
            lora_id: None,
            from_medium: "GPU".into(),
            to_medium: "CPU".into(),
        });
        log.publish(KvEvent::AllBlocksCleared);
        let batch = log.drain(Some(0)).unwrap();
        let v = batch.to_vllm();
        assert_eq!(v[2], json!(0));
        let events = v[1].as_array().unwrap();
        assert_eq!(events.len(), 4);
        assert_eq!(
            events[0],
            json!(["BlockStored", [1, 2], null, [5, 6, 7, 8], 2, null, "GPU"])
        );
        assert_eq!(events[1], json!(["BlockRemoved", [1], "GPU"]));
        assert_eq!(events[2][0], json!("BlockStored"));
        assert_eq!(events[2][6], json!("CPU"));
        assert_eq!(events[3], json!(["AllBlocksCleared"]));
        // Native JSON form round-trips (fixed timestamp: serde_json float parsing may differ
        // from the printed value in the last bit).
        let batch = KvEventBatch { ts: 1.5, ..batch };
        let s = serde_json::to_string(&batch).unwrap();
        assert_eq!(serde_json::from_str::<KvEventBatch>(&s).unwrap(), batch);
    }
}
