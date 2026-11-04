from typing import Any, Callable, Optional, Sequence, Union

WIRE_VERSION: int
LAYOUT_VERSION: int
MEMORY_KINDS: list[str]
REPRESENTATIONS: list[str]

class Region:
    @staticmethod
    def host(nbytes: int) -> Region: ...
    @staticmethod
    def from_buffer(obj: Any, location: str = "host") -> Region: ...
    @staticmethod
    def external(location: str, addr: int, nbytes: int, obj: Any) -> Region: ...
    @property
    def kind(self) -> str: ...
    @property
    def location(self) -> str: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def addr(self) -> int: ...
    @property
    def state(self) -> str: ...
    def read(self, offset: int = 0, nbytes: Optional[int] = None) -> bytes: ...
    def write(self, offset: int, data: bytes) -> None: ...

class HostEngine:
    def __init__(self) -> None: ...
    @property
    def capabilities(self) -> list[tuple[str, str, str, str]]: ...

class RegionInfo:
    kind: str
    location: str
    index: int
    addr: int
    nbytes: int
    obj: Any

class CopyJob:
    src: list[RegionInfo]
    dst: list[RegionInfo]
    spans: list[tuple[int, int, int, int, int]]
    representation: str
    dst_representation: str
    direction: str
    nbytes: int

class Route:
    def __init__(
        self,
        engine: Any,
        src: str = "host",
        dst: str = "host",
        representation: str = "bf16",
        dst_representation: Optional[str] = None,
    ) -> None: ...
    src: str
    dst: str
    representation: str
    dst_representation: str
    engine: str

class Layout:
    def __init__(
        self,
        representation: str,
        layers: int,
        block_size: int,
        heads: int,
        head_dim: int,
        *,
        blocks: int = 1,
        kv: str = "kv",
        order: Optional[Sequence[str]] = None,
        split: str = "none",
        strides: Optional[dict[str, int]] = None,
        region_bytes: Optional[int] = None,
    ) -> None: ...
    @staticmethod
    def preset(
        name: str,
        representation: str,
        layers: int,
        block_size: int,
        heads: int,
        head_dim: int,
        *,
        blocks: int = 1,
        kv: str = "kv",
    ) -> Layout: ...
    version: int
    representation: str
    kv: str
    layers: int
    blocks: int
    block_size: int
    heads: int
    head_dim: int
    split: str
    strides: dict[str, int]
    region_bytes: int
    region_count: int

class KVSpec:
    def __init__(self, model_id: str, revision: str, layout: Layout) -> None: ...
    model_id: str
    revision: str
    layout: Layout
    representation: str
    def check_compatible(self, other: KVSpec) -> None: ...

class KVBuffer:
    def __init__(self, spec: KVSpec, regions: list[Region]) -> None: ...
    spec: KVSpec
    kind: str
    location: str
    def phases(self) -> list[str]: ...
    def reclaim(self) -> list[Region]: ...

class BufferRef:
    spec: KVSpec
    kind: str
    location: str

Buffer = Union[KVBuffer, BufferRef]

class KVDescriptor:
    def to_bytes(self) -> bytes: ...
    @staticmethod
    def from_bytes(data: bytes) -> KVDescriptor: ...
    spec: KVSpec
    agent_id: Optional[int]
    block_hashes: dict[int, Union[int, bytes]]
    def resolve(self) -> BufferRef: ...

class LayerProgress:
    def __init__(self, layers: int, start: int, end: int) -> None: ...
    def publish(self, layer_start: int, layer_end: int, token_start: int, token_end: int) -> None: ...
    def drain(self) -> list[tuple[int, int, int, int]]: ...
    def finish(self) -> None: ...
    pending: bool
    finished: bool
    layers: int
    start: int
    end: int

class Session:
    def __init__(
        self,
        engine: Union[Route, Any],
        src: str = "host",
        dst: str = "host",
        representation: str = "bf16",
        dst_representation: Optional[str] = None,
    ) -> None: ...
    state: str
    request_id: Optional[str]
    poison_reason: Optional[str]
    retained: int
    engine: str
    def reserve(self, request_id: str, dst: Union[Buffer, list[Buffer]]) -> None: ...
    def transfer(
        self, request_id: str, src: Buffer, selection: Any, *, group: int = 0, direction: str = "write"
    ) -> int: ...
    def poll(self, request_id: str, transfer: int) -> bool: ...
    def wait(self, request_id: str, transfer: int) -> None: ...
    def wait_layer(self, request_id: str, layer: int) -> None: ...
    def ready_layers(self, request_id: str) -> list[int]: ...
    def fence(self, request_id: str) -> dict[str, Any]: ...
    def admit(self, request_id: str, notify: Optional[Callable[[], Any]] = None) -> dict[str, Any]: ...
    def finish(self, request_id: str, passed: bool = True) -> None: ...
    def cancel(self, request_id: str) -> None: ...
    def recover(self) -> None: ...

class Agent:
    def __init__(self, name: str, endpoint: Optional[str] = None) -> None: ...
    id: int
    name: str
    def meta(self) -> bytes: ...
    def export(self, buffer: Buffer, block_hashes: Optional[dict[int, Union[int, bytes]]] = None) -> KVDescriptor: ...
    def release(self, descriptor: KVDescriptor) -> None: ...
    def resolve(self, descriptor: KVDescriptor) -> BufferRef: ...
    def publish(self, event: dict[str, Any]) -> None: ...
    def drain_events(self, data_parallel_rank: Optional[int] = None, vllm: bool = True) -> Any: ...

def decode_agent_meta(data: bytes) -> dict[str, Any]: ...
def plan(src: KVSpec, dst: KVSpec, selection: Any) -> list[tuple[int, int, int, int, int]]: ...
def canonical_representation(name: str) -> str: ...
