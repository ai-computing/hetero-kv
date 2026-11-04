"""Skeleton: a vLLM V1 KV connector on hetero-kv (not functional in P0).

Shows where each ``KVConnectorBase_V1`` hook (names as of vLLM v0.10/v0.11; check your pin) maps
onto hetero-kv. A real connector subclasses
``vllm.distributed.kv_transfer.kv_connector.v1.base.KVConnectorBase_V1``; this file does not
import vLLM so it stays importable in hetero-kv's CI.

Missing before this can work: cross-process descriptor resolution and notifications (P3), and a
device engine for vLLM's GPU tensors (CUDA, P1; the Torch engine can stand in on one host).
"""

from __future__ import annotations

import hetero_kv as hk


class HeteroKVConnector:
    """P/D connector. Decode side pulls (``direction="read"``, like NixlConnector); the producer
    can also push (``"write"``) when the decode side exported its blocks first."""

    def __init__(self, engine_id: str, engine, representation: str, src_kind="cuda", dst_kind="cuda"):
        self.agent = hk.Agent(engine_id)
        self.session_args = (engine, src_kind, dst_kind, representation)
        self.buffer: hk.KVBuffer | None = None
        self.sessions: dict[str, hk.Session] = {}

    # ---- worker side --------------------------------------------------------------------
    def register_kv_caches(self, kv_caches: dict) -> None:
        """vLLM hands per-layer tensors ``[2, num_blocks, block_size, heads, head_dim]``:
        ``Layout.preset("layer_separate", ...)`` with one region per layer
        (``hetero_kv.torch_engine.tensor_region``), then ``agent.export`` to obtain the descriptor
        that goes into ``kv_transfer_params``."""
        raise NotImplementedError

    def start_load_kv(self, forward_context, **kwargs) -> None:
        """For each request with remote blocks: resolve the peer descriptor, ``reserve`` the local
        blocks, and ``transfer(Selection.blocks(remote_ids, local_ids, layers=(l, l + 1)),
        direction="read")`` per layer so ``wait_for_layer_load`` can wait per layer."""
        raise NotImplementedError

    def wait_for_layer_load(self, layer_name: str) -> None:
        """``session.wait_layer(request_id, layer_index)`` for every loading request."""
        raise NotImplementedError

    def save_kv_layer(self, layer_name: str, kv_layer, attn_metadata, **kwargs) -> None:
        """Push mode only: ``transfer(..., direction="write")`` for this layer into the remote
        destination descriptor."""
        raise NotImplementedError

    def wait_for_save(self) -> None:
        """``session.fence(request_id)`` for every request pushed this step."""
        raise NotImplementedError

    def get_finished(self, finished_req_ids: set[str]):
        """``ready_layers``/``fence`` decide which loads finished; ``admit`` then lets the
        scheduler run them; ``finish`` after the request completes."""
        raise NotImplementedError

    # ---- scheduler side -----------------------------------------------------------------
    def request_finished(self, request, block_ids: list[int]):
        """Prefill side: return ``kv_transfer_params`` = {"remote_engine_id": agent id,
        "remote_block_ids": block_ids, "hetero_kv_descriptor": descriptor bytes (base64),
        "remote_host"/"remote_port": AgentMeta.endpoint}. The blocks stay pinned until the decode
        side reports completion (P3 notification)."""
        raise NotImplementedError
