"""Skeleton: hetero-kv as the transport under a KV manager (LMCache backend, Mooncake, KVBM).

Not functional in P0: the storage and remote tiers have no engine yet. The shape is what matters:
a KV manager keeps its own index and policy, and uses hetero-kv to (1) describe block buffers per
tier, (2) move blocks between tiers with the same session/engine API as P/D, and (3) publish
block events that a KV-cache-aware router (llm-d) consumes.
"""

from __future__ import annotations

import hetero_kv as hk


class TieredBlockStore:
    """Blocks addressed by content hash across tiers (device -> host_pinned -> storage/remote)."""

    def __init__(self, agent: hk.Agent, tiers: dict[str, hk.KVBuffer], engines: dict[tuple[str, str], object]):
        # tiers: memory kind -> block-major KVBuffer (Layout.preset("block_major", ...)),
        # engines: (src kind, dst kind) -> engine serving that route. No fallback: a missing
        # route is an UnsupportedError when the session is created.
        self.agent, self.tiers, self.engines = agent, tiers, engines
        self.index: dict[int, tuple[str, int]] = {}   # block hash -> (tier, block id)

    def contains(self, block_hash: int) -> bool:
        """LMCache ``contains`` / Mooncake ``isExist``: the manager's own index."""
        return block_hash in self.index

    def move(self, block_hashes: list[int], dst_tier: str, dst_blocks: list[int]) -> None:
        """Offload/onload: one session per route, ``Selection.blocks(src_ids, dst_ids)``, then
        publish ``BlockMoved`` so routers see the new medium. Needs storage/remote engines."""
        raise NotImplementedError

    def put(self, block_hash: int, tier: str, block: int, token_ids: list[int], parent=None) -> None:
        """Record a stored block and emit the vLLM-shaped ``BlockStored`` event."""
        self.index[block_hash] = (tier, block)
        self.agent.publish({
            "type": "BlockStored", "block_hashes": [block_hash], "parent_block_hash": parent,
            "token_ids": token_ids, "block_size": self.tiers[tier].spec.layout.block_size,
            "lora_id": None, "medium": tier.upper(),
        })

    def remove(self, block_hash: int) -> None:
        tier, _ = self.index.pop(block_hash)
        self.agent.publish({"type": "BlockRemoved", "block_hashes": [block_hash], "medium": tier.upper()})
