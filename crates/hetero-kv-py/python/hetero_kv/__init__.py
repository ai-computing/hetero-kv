"""hetero-kv: layered KV-cache transport between heterogeneous accelerators.

L1 ``Region`` · L2 ``Route``, ``HostEngine`` / Python engines · L3 ``Layout``, ``KVSpec``, ``KVBuffer``,
``KVDescriptor``, ``Selection``, ``LayerProgress`` · L4 ``Session``, ``Agent``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from ._errors import (
    BusyError,
    CancelledError,
    HeteroKVError,
    InvalidError,
    PoisonedError,
    StateError,
    TransferError,
    UnsupportedError,
)
from ._native import (
    LAYOUT_VERSION,
    MEMORY_KINDS,
    REPRESENTATIONS,
    WIRE_VERSION,
    Agent,
    BufferRef,
    CopyJob,
    HostEngine,
    KVBuffer,
    KVDescriptor,
    KVSpec,
    LayerProgress,
    Layout,
    Region,
    RegionInfo,
    Route,
    Session,
    canonical_representation,
    decode_agent_meta,
    plan,
)

__version__ = "0.1.0"


@dataclass(frozen=True)
class Selection:
    """Which KV to copy: ``layers`` (None = all) and a token range addressed through block
    tables. Logical source token ``src_start + i`` maps to destination token ``dst_start + i``
    for ``i < tokens`` (None = rest of the source table). A contiguous cache is one block."""

    layers: Optional[Tuple[int, int]] = None
    src_blocks: Sequence[int] = (0,)
    dst_blocks: Sequence[int] = (0,)
    src_start: int = 0
    dst_start: int = 0
    tokens: Optional[int] = None

    @classmethod
    def prefix(cls, tokens: int, layers: Optional[Tuple[int, int]] = None) -> "Selection":
        """The first ``tokens`` tokens of contiguous caches."""
        return cls(layers=layers, tokens=tokens)

    @classmethod
    def contiguous(
        cls, src_start: int, dst_start: int, tokens: int, layers: Optional[Tuple[int, int]] = None
    ) -> "Selection":
        """A token range of contiguous caches (e.g. one layer-progress span)."""
        return cls(layers=layers, src_start=src_start, dst_start=dst_start, tokens=tokens)

    @classmethod
    def blocks(
        cls, src_ids: Sequence[int], dst_ids: Sequence[int], layers: Optional[Tuple[int, int]] = None
    ) -> "Selection":
        """Whole paged blocks ``src_ids[i] -> dst_ids[i]`` (vLLM block ids)."""
        return cls(layers=layers, src_blocks=tuple(src_ids), dst_blocks=tuple(dst_ids))


def transfer_and_admit(session: Session, request_id: str, src, dst, selections, notify=None,
                       *, direction: str = "write") -> dict:
    """reserve -> transfer -> fence -> admit in one call: returns only after verified completion
    and destination visibility (the Optimus ``Transport.transfer`` contract)."""
    session.reserve(request_id, dst)
    session.transfer(request_id, src, selections, direction=direction)
    session.fence(request_id)
    return session.admit(request_id, notify)


__all__ = [
    "Agent", "BufferRef", "BusyError", "CancelledError", "CopyJob", "HeteroKVError", "HostEngine",
    "InvalidError", "KVBuffer", "KVDescriptor", "KVSpec", "LAYOUT_VERSION", "LayerProgress",
    "Layout", "MEMORY_KINDS", "PoisonedError", "REPRESENTATIONS", "Region", "RegionInfo",
    "Route", "Selection", "Session", "StateError", "TransferError", "UnsupportedError", "WIRE_VERSION",
    "canonical_representation", "decode_agent_meta", "plan", "transfer_and_admit",
]
