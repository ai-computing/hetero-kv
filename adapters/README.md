# Adapters

Thin layers that map a consumer's integration API onto hetero-kv. They depend on hetero-kv;
hetero-kv never depends on them. Glue that needs a consumer's internals belongs in the consumer
(Optimus Prime keeps its own); the skeletons here use only documented consumer APIs.

| Directory | Consumer | P0 status |
|---|---|---|
| — | aicomp optimus-rm `opt_prime/accel/pd` | Owned by Optimus: `accel/pd/transport.py` (not here) |
| [`vllm/`](vllm/) | vLLM `KVConnectorBase_V1` (and llm-d P/D through it) | Skeleton + design notes |
| [`kvstore/`](kvstore/) | LMCache storage backend, Mooncake transfer engine / store, Dynamo KVBM | Skeleton + design notes |
| [`nixl/`](nixl/) | NIXL backend plugin (C++ over the C ABI) | Design notes |

See [INTEGRATION.md](../INTEGRATION.md) for the concept mapping and the gaps.
