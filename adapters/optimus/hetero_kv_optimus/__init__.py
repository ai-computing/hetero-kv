"""Optimus Prime (aicomp optimus-rm ``opt_prime/accel/pd``) adapter for hetero-kv.

Drop-in replacements, duck-typed against Optimus' contracts (``KVLease`` with ``spec``,
``endpoint`` and ``payload``; the ``Transport`` protocol ``supports``/``transfer``/``finish``
plus ``recover``). Nothing here imports ``opt_prime``.

=========================================  ==============================================
Optimus today                              hetero-kv
=========================================  ==============================================
``layer_progress.LayerProgress``           ``LayerProgress`` (= ``hetero_kv.LayerProgress``)
``torch_cache.TorchCacheTransport``        ``TorchCacheTransport`` (session + TorchCopyEngine)
``hetero_p2p.HeteroP2PTransport``          ``SessionTransport`` (any engine + plan builder)
=========================================  ==============================================
"""

from .transport import LayerProgress, SessionTransport, TorchCacheTransport, TransferPlan

__all__ = ["LayerProgress", "SessionTransport", "TorchCacheTransport", "TransferPlan"]
