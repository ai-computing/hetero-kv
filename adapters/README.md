# Adapters

Thin layers that map a consumer's integration API onto hetero-kv. They depend on hetero-kv;
hetero-kv never depends on them.

| Directory | Consumer | P0 status |
|---|---|---|
| [`optimus/`](optimus/) | aicomp optimus-rm `opt_prime/accel/pd` (`Transport`, `LayerProgress`) | **Real and tested** (`hetero_kv_optimus`) |
| [`vllm/`](vllm/) | vLLM `KVConnectorBase_V1` (and llm-d P/D through it) | Skeleton + design notes |
| [`kvstore/`](kvstore/) | LMCache storage backend, Mooncake transfer engine / store, Dynamo KVBM | Skeleton + design notes |
| [`nixl/`](nixl/) | NIXL backend plugin (C++ over the C ABI) | Design notes |

See [INTEGRATION.md](../INTEGRATION.md) for the concept mapping and the gaps.
