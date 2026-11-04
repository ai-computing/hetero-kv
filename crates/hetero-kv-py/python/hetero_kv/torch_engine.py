"""A copy engine implemented with Torch ``copy_`` (host <-> host/CUDA), and tensor regions.

This lets a Torch program use the hetero-kv session lifecycle before the native CUDA engine
(phase P1) exists. Torch is imported lazily; importing ``hetero_kv`` never imports it.
"""

from __future__ import annotations

from typing import Any, Iterable

from ._native import CopyJob, Region


def tensor_location(device: Any) -> str:
    """hetero-kv location of a torch device: ``host`` or ``cuda:<index>``."""
    if device.type == "cpu":
        return "host"
    if device.type == "cuda":
        import torch

        return f"cuda:{device.index if device.index is not None else torch.cuda.current_device()}"
    raise ValueError(f"no memory location for device {device}")


def tensor_region(tensor: Any) -> Region:
    """Wrap a contiguous tensor's storage as a region at ``host`` or ``cuda:<index>``. The
    tensor stays alive as long as the region and is what the engine receives as
    ``RegionInfo.obj``."""
    if not tensor.is_contiguous():
        raise ValueError("tensor regions must be contiguous")
    location = tensor_location(tensor.device)
    return Region.external(location, tensor.data_ptr(), tensor.numel() * tensor.element_size(), tensor)


def _bytes(info):
    import torch

    t = info.obj
    if t is None or not isinstance(t, torch.Tensor):
        raise ValueError("TorchCopyEngine needs tensor-backed regions (see tensor_region)")
    return t.reshape(-1).view(torch.uint8)


class TorchCopyEngine:
    """Copies spans with ``Tensor.copy_`` on the current stream. A CUDA job completes when an
    event recorded after its copies completes; a CPU-only job completes at submit."""

    name = "torch-copy"

    def __init__(self, kinds: Iterable[str] = ("host", "cuda")):
        kinds = tuple(kinds)
        # Every device of each kind; byte copies are representation-agnostic (None = all
        # representations, identity only: copy_ on uint8 views never converts).
        self.capabilities = [(s, d, None) for s in kinds for d in kinds]

    def submit(self, job: CopyJob):
        import torch

        src = [_bytes(r) for r in job.src]
        dst = [_bytes(r) for r in job.dst]
        with torch.inference_mode():
            for si, so, di, do, n in job.spans:
                dst[di][do:do + n].copy_(src[si][so:so + n], non_blocking=True)
        devices = {t.device for t in src + dst if t.device.type == "cuda"}
        events = []
        for device in devices:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
            events.append(event)
        return events, job.nbytes

    def poll(self, ticket) -> bool:
        events, _ = ticket
        return all(e.query() for e in events)

    def wait(self, ticket) -> int:
        events, nbytes = ticket
        for e in events:
            e.synchronize()
        return nbytes

    def quiesce(self) -> bool:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
        return True
