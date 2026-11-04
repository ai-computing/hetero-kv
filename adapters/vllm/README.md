# vLLM KV connector (skeleton)

`hetero_kv_vllm.py` maps each `KVConnectorBase_V1` hook to hetero-kv calls; every method raises
`NotImplementedError` in P0. The mapping table and the gaps are in
[INTEGRATION.md](../../INTEGRATION.md#vllm-kv-connector). It becomes functional once
cross-process descriptors and notifications (P3) and the CUDA engine (P1) exist.
