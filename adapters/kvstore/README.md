# KV managers: LMCache, Mooncake, Dynamo KVBM (skeleton)

`hetero_kv_kvstore.py` sketches a tiered block store that keeps its own index and uses hetero-kv
for block-major buffers per tier, tier-to-tier block moves (`Selection.blocks`) and vLLM-shaped
block events (`Agent.publish` / `drain_events`). The event part works today; block moves to
storage or remote tiers need engines that do not exist in P0. See
[INTEGRATION.md](../../INTEGRATION.md#kv-managers-lmcache-mooncake-dynamo-kvbm).
