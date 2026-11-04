"""Optimus ``Transport`` implementations on the hetero-kv session lifecycle.

Behaviour contract kept from Optimus:

* ``supports(source, destination, spec)`` matches only the configured route (no fallback).
* ``transfer`` returns only after verified completion and destination visibility; it raises
  ``ValueError`` for an unconfigured route or mismatched physical format, before any copy.
* ``finish(request_id)`` ends the decode lease; ``recover()`` returns True once the engine is
  quiesced and the lane may be reused, False otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import hetero_kv as hk

LayerProgress = hk.LayerProgress


def _settle(session: hk.Session) -> bool:
    """Bring a session back to idle after the caller's lane failed. True on success."""
    rid = session.request_id
    try:
        if session.state == "admitted":
            try:
                session.finish(rid, passed=False)  # the decoder failed while using the cache
            except hk.PoisonedError:
                pass                               # expected: a failed consumer poisons
        elif session.state in ("reserved", "fenced"):
            session.cancel(rid)
        elif session.state == "transferring":
            session.cancel(rid)
            session.fence(rid)                     # never abandon in-flight copies
            try:
                session.admit(rid)
            except hk.CancelledError:
                pass
        if session.state == "poisoned":
            session.recover()
    except hk.HeteroKVError:
        return False
    return session.state == "idle"


@dataclass(frozen=True)
class TransferPlan:
    """What a plan builder returns: the source buffer, one destination buffer per KV group, and
    the selections to copy into each group."""

    source: Any                                   # hk.KVBuffer | hk.BufferRef
    destinations: Sequence[Any]
    selections: Sequence[Sequence[hk.Selection]]


class SessionTransport:
    """Replacement for ``hetero_p2p.HeteroP2PTransport``: an explicit route over any hetero-kv
    engine. ``plan_builder(source_lease, destination_lease) -> TransferPlan`` must use live,
    owned allocations; ``notify(destination_lease)`` makes the fenced cache visible to the
    decoder. Session states map to the old ``CopySession`` ones as fenced = ready,
    admitted = decoding."""

    def __init__(self, engine, plan_builder: Callable[[Any, Any], TransferPlan], notify: Callable[[Any], Any],
                 *, source: str, destination: str, spec, src_kind: str = "host", dst_kind: str = "host"):
        self.representation = hk.canonical_representation(spec.representation)
        self.session = hk.Session(engine, src_kind, dst_kind, self.representation)
        self.plan_builder, self.notify = plan_builder, notify
        self.source, self.destination, self.spec = source, destination, spec

    @property
    def state(self) -> str:
        return self.session.state

    def supports(self, source, destination, spec) -> bool:
        return (source, destination, spec) == (self.source, self.destination, self.spec)

    def transfer(self, request_id, source, destination):
        if source.spec != destination.spec or not self.supports(source.endpoint, destination.endpoint, source.spec):
            raise ValueError("unsupported P2P route or physical format")
        plan = self.plan_builder(source, destination)
        buffers = [plan.source, *plan.destinations]
        if any(b.spec.representation != self.representation for b in buffers):
            raise ValueError("DMA plan format differs from negotiated representation")
        if len(plan.selections) != len(plan.destinations):
            raise ValueError("one selection list per destination group required")
        s = self.session
        s.reserve(request_id, list(plan.destinations))
        try:
            for group, selections in enumerate(plan.selections):
                if selections:
                    s.transfer(request_id, plan.source, list(selections), group=group)
            s.fence(request_id)
            return s.admit(request_id, lambda: self.notify(destination))
        except (hk.InvalidError, hk.UnsupportedError, hk.StateError):
            _settle(s)                             # rejected before the engine saw anything
            raise

    def finish(self, request_id):
        self.session.finish(request_id, passed=True)

    def recover(self) -> bool:
        return _settle(self.session)


def _layer_tensors(manager):
    """(K, V) tensors of each local layer of an Optimus KVCacheManager or LayeredKVCache."""
    layers = range(manager.layer_start, manager.layer_end)
    if hasattr(manager, "k") and hasattr(manager, "v"):          # LayeredKVCache
        return [(manager.k[i], manager.v[i]) for i in layers]
    return [manager._cache[i] for i in layers]                    # KVCacheManager [B, H, S, D]


def _groups(manager, spec):
    """KV buffers over the manager's tensors: one bhsd group, or one per layer when layer shapes
    differ (e.g. Gemma 4). Blocks are batch entries; block_size is the sequence capacity."""
    from hetero_kv.torch_engine import tensor_region

    pairs = _layer_tensors(manager)
    shapes = {tuple(k.shape) for k, _ in pairs}
    chunks = [pairs] if len(shapes) == 1 else [[p] for p in pairs]
    groups = []
    for chunk in chunks:
        batch, heads, capacity, dim = chunk[0][0].shape
        layout = hk.Layout.preset("bhsd", str(chunk[0][0].dtype), len(chunk), capacity, heads, dim, blocks=batch)
        regions = [tensor_region(t) for pair in chunk for t in pair]
        groups.append(hk.KVBuffer(hk.KVSpec(spec.model_id, spec.revision, layout), regions))
    return groups, chunks[0][0][0]


_KIND = {"cpu": "host", "cuda": "cuda"}


class TorchCacheTransport:
    """Replacement for ``torch_cache.TorchCacheTransport``: explicit tensor-copy transport,
    never selected as a P2P fallback. Copies the source prefix ``[0, seq_len)`` of every layer
    (and batch entry) with ``TorchCopyEngine``, fences, then publishes ``seq_len`` on the
    destination as the admission step."""

    def __init__(self, *, source, destination, spec):
        self.source, self.destination, self.spec = source, destination, spec
        self._sessions: dict[tuple, hk.Session] = {}
        self._active: dict[str, tuple] = {}

    def supports(self, source, destination, spec):
        return (source, destination, spec) == (self.source, self.destination, self.spec)

    def _session(self, src_kind, dst_kind, representation):
        from hetero_kv.torch_engine import TorchCopyEngine

        key = (src_kind, dst_kind, representation)
        if key not in self._sessions:
            self._sessions[key] = hk.Session(TorchCopyEngine(), *key)
        return self._sessions[key]

    def transfer(self, request_id, source, destination):
        if source.spec != destination.spec or not self.supports(source.endpoint, destination.endpoint, source.spec):
            raise ValueError("unconfigured tensor-copy route")
        src, dst = source.payload.manager, destination.payload.manager
        position = src.get_seq_len()
        if not 0 < position <= dst.max_seq_len:
            raise ValueError("invalid source sequence length")
        src_groups, src_t = _groups(src, source.spec)
        dst_groups, dst_t = _groups(dst, destination.spec)
        if len(src_groups) != len(dst_groups):
            raise ValueError("source and destination layer groups differ")
        try:
            kinds = (_KIND[src_t.device.type], _KIND[dst_t.device.type])
        except KeyError as error:
            raise ValueError(f"no tensor-copy route for device {error}") from None
        session = self._session(*kinds, hk.canonical_representation(str(src_t.dtype)))
        batch = src_t.shape[0]
        session.reserve(request_id, dst_groups)
        try:
            for group, buf in enumerate(src_groups):
                selections = [hk.Selection(src_blocks=(b,), dst_blocks=(b,), tokens=position) for b in range(batch)]
                session.transfer(request_id, buf, selections, group=group)
            receipt = session.fence(request_id)
            session.admit(request_id, lambda: dst.set_seq_len(position))
        except (hk.InvalidError, hk.UnsupportedError, hk.StateError):
            _settle(session)
            raise
        self._active[request_id] = (session, src_groups, dst_groups)
        return dict(transport="torch-copy", bytes=receipt["bytes"], cache_position=position, completed=True)

    def finish(self, request_id):
        active = self._active.pop(request_id, None)
        if active is not None:
            active[0].finish(request_id, passed=True)

    def recover(self):
        """Quiesce (Torch device sync) and settle every session; True when all are idle."""
        ok = all([_settle(s) for s in self._sessions.values()])
        if ok:
            self._active.clear()
        return ok
