"""Exception hierarchy. Rejections (nothing was submitted) derive from ValueError or are
StateError/BusyError; TransferError means buffers are uncertain and the session is poisoned."""


class HeteroKVError(Exception):
    """Base class of every hetero-kv error."""


class InvalidError(HeteroKVError, ValueError):
    """Malformed argument; nothing was submitted."""


class UnsupportedError(HeteroKVError, ValueError):
    """No route, capability or phase supports this; there is no fallback."""


class StateError(HeteroKVError, RuntimeError):
    """Wrong lifecycle step or stale request; nothing changed."""


class BusyError(HeteroKVError, RuntimeError):
    """The single-flight object is held by another request, or memory is still referenced."""


class CancelledError(HeteroKVError, RuntimeError):
    """Cancellation completed after the copies were fenced."""


class PoisonedError(HeteroKVError, RuntimeError):
    """The session is poisoned; call recover()."""


class TransferError(HeteroKVError, RuntimeError):
    """An engine failed or reported an unverifiable completion; buffers are uncertain."""
