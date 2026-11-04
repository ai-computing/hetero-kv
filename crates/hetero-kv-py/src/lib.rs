//! Python bindings (`hetero_kv._native`). The public Python API is re-exported by the pure-Python
//! package `hetero_kv`; see `python/hetero_kv/_native.pyi` for signatures.
//!
//! Every call that takes a session lock or may wait on an engine first releases the GIL, so a
//! Python engine (which needs the GIL) can never deadlock against another Python thread.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyInt, PyList, PyType};

use hetero_kv::engine::{
    Capabilities, Completion, CopyEngine, CopyJob, DevicePattern, Direction, HostEngine,
    Representation, Route,
};
use hetero_kv::kv::{
    self, Axis, BlockHash, Geometry, KvBuffer, KvBufferRef, KvDescriptor, KvEvent, KvKind, KvSpec,
    Layout, Selection, Split, Strides,
};
use hetero_kv::memory::{Location, MemoryKind, Region};
use hetero_kv::session::{self as sess, AgentMeta, Receipt};
use hetero_kv::Error;

// ---------------------------------------------------------------------------------------------
// Errors

fn to_py(e: Error) -> PyErr {
    let name = match &e {
        Error::Invalid(_) => "InvalidError",
        Error::Unsupported(_) => "UnsupportedError",
        Error::State(_) => "StateError",
        Error::Busy(_) => "BusyError",
        Error::Cancelled(_) => "CancelledError",
        Error::Poisoned(_) => "PoisonedError",
        Error::Transfer(_) => "TransferError",
    };
    Python::attach(|py| {
        match py
            .import("hetero_kv._errors")
            .and_then(|m| m.getattr(name))
            .and_then(|c| Ok(c.cast_into::<PyType>()?))
        {
            Ok(cls) => PyErr::from_type(cls, e.to_string()),
            Err(err) => err,
        }
    })
}

trait IntoPy<T> {
    fn py(self) -> PyResult<T>;
}
impl<T> IntoPy<T> for hetero_kv::Result<T> {
    fn py(self) -> PyResult<T> {
        self.map_err(to_py)
    }
}

fn parse<T: std::str::FromStr<Err = Error>>(s: &str) -> PyResult<T> {
    s.parse().py()
}

// ---------------------------------------------------------------------------------------------
// L1: regions

/// Keeps a Python object (and an exported buffer view) alive for an external region.
struct PyKeep {
    obj: Py<PyAny>,
    _view: Option<PyBuffer<u8>>,
}

/// A memory region. Moves into a `KVBuffer`; get it back with `KVBuffer.reclaim()`.
#[pyclass(name = "Region", module = "hetero_kv", frozen)]
struct PyRegion {
    inner: Mutex<Option<Region>>,
}

impl PyRegion {
    fn wrap(r: Region) -> Self {
        PyRegion {
            inner: Mutex::new(Some(r)),
        }
    }
    fn with<T>(&self, f: impl FnOnce(&mut Region) -> PyResult<T>) -> PyResult<T> {
        let mut g = self.inner.lock().unwrap();
        match g.as_mut() {
            Some(r) => f(r),
            None => Err(to_py(Error::State(
                "region was moved into a KVBuffer".into(),
            ))),
        }
    }
    fn take(&self) -> PyResult<Region> {
        self.inner
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| to_py(Error::State("region was moved into a KVBuffer".into())))
    }
}

#[pymethods]
impl PyRegion {
    /// Allocate zeroed, page-aligned host memory.
    #[staticmethod]
    fn host(nbytes: usize) -> PyResult<Self> {
        Ok(Self::wrap(Region::host(nbytes).py()?))
    }

    /// Wrap a writable, C-contiguous buffer (bytearray, numpy array, mmap, ...) as host memory
    /// (`location` "host" or "host_pinned").
    #[staticmethod]
    #[pyo3(signature = (obj, location = "host"))]
    fn from_buffer(obj: &Bound<'_, PyAny>, location: &str) -> PyResult<Self> {
        let location: Location = parse(location)?;
        if !location.kind.is_host() || location.node.is_some() {
            return Err(PyValueError::new_err(
                "from_buffer wraps local host memory only",
            ));
        }
        let view = PyBuffer::<u8>::get(obj)?;
        if view.readonly() || !view.is_c_contiguous() {
            return Err(PyValueError::new_err(
                "buffer must be writable and C-contiguous",
            ));
        }
        let (addr, len) = (view.buf_ptr() as usize, view.len_bytes());
        let keep = PyKeep {
            obj: obj.clone().unbind(),
            _view: Some(view),
        };
        // SAFETY: the exported buffer view pins `addr..addr+len` while the region lives; the
        // caller must not use the object's memory while the region is lent.
        let r = unsafe { Region::external(location, addr, len, Some(Box::new(keep))) }.py()?;
        Ok(Self::wrap(r))
    }

    /// Wrap memory at `location` ("host", "cuda:1", "rngd:3", ...) described by an address (e.g.
    /// `tensor.data_ptr()`); `obj` keeps it alive and is handed to engines as `RegionInfo.obj`.
    /// The caller vouches for the address.
    #[staticmethod]
    fn external(location: &str, addr: usize, nbytes: usize, obj: Py<PyAny>) -> PyResult<Self> {
        let location: Location = parse(location)?;
        let keep = PyKeep { obj, _view: None };
        // SAFETY: documented caller contract (see the Python docstring).
        let r = unsafe { Region::external(location, addr, nbytes, Some(Box::new(keep))) }.py()?;
        Ok(Self::wrap(r))
    }

    #[getter]
    fn kind(&self) -> PyResult<&'static str> {
        self.with(|r| Ok(r.kind().as_str()))
    }
    /// "kind:index[@node]".
    #[getter]
    fn location(&self) -> PyResult<String> {
        self.with(|r| Ok(r.location().to_string()))
    }
    #[getter]
    fn nbytes(&self) -> PyResult<usize> {
        self.with(|r| Ok(r.len()))
    }
    #[getter]
    fn addr(&self) -> PyResult<usize> {
        self.with(|r| Ok(r.addr()))
    }
    /// "owned" or "moved".
    #[getter]
    fn state(&self) -> &'static str {
        if self.inner.lock().unwrap().is_some() {
            "owned"
        } else {
            "moved"
        }
    }

    /// Copy host bytes out.
    #[pyo3(signature = (offset = 0, nbytes = None))]
    fn read<'py>(
        &self,
        py: Python<'py>,
        offset: usize,
        nbytes: Option<usize>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.with(|r| {
            let s = r
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("not host memory"))?;
            let end = nbytes.map_or(Some(s.len()), |n| offset.checked_add(n));
            match end.and_then(|e| s.get(offset..e)) {
                Some(b) => Ok(PyBytes::new(py, b)),
                None => Err(PyValueError::new_err("range outside region")),
            }
        })
    }

    /// Copy host bytes in.
    fn write(&self, offset: usize, data: &[u8]) -> PyResult<()> {
        self.with(|r| {
            let s = r
                .as_mut_slice()
                .ok_or_else(|| PyValueError::new_err("not host memory"))?;
            match offset
                .checked_add(data.len())
                .and_then(|e| s.get_mut(offset..e))
            {
                Some(d) => {
                    d.copy_from_slice(data);
                    Ok(())
                }
                None => Err(PyValueError::new_err("range outside region")),
            }
        })
    }

    fn __repr__(&self) -> String {
        match self.inner.lock().unwrap().as_ref() {
            Some(r) => format!("Region({}, {} bytes)", r.kind(), r.len()),
            None => "Region(moved)".into(),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// L2: engines

/// The host reference engine (synchronous memmove between host regions).
#[pyclass(name = "HostEngine", module = "hetero_kv", frozen)]
struct PyHostEngine {
    engine: Arc<HostEngine>,
}

#[pymethods]
impl PyHostEngine {
    #[new]
    fn new() -> Self {
        PyHostEngine {
            engine: Arc::new(HostEngine::new()),
        }
    }
    /// Served `(src_devices, dst_devices, src_repr, dst_repr)` routes.
    #[getter]
    fn capabilities(&self) -> Vec<(String, String, &'static str, &'static str)> {
        self.engine
            .capabilities()
            .routes()
            .iter()
            .map(|c| {
                (
                    c.src.to_string(),
                    c.dst.to_string(),
                    c.src_repr.as_str(),
                    c.dst_repr.as_str(),
                )
            })
            .collect()
    }
}

/// `(src_index, src_offset, dst_index, dst_offset, nbytes)`.
type SpanTuple = (u32, u64, u32, u64, u64);

/// Region as seen by a Python engine.
#[pyclass(name = "RegionInfo", module = "hetero_kv", frozen, get_all)]
struct RegionInfo {
    kind: &'static str,
    /// "kind:index[@node]".
    location: String,
    /// Device index within the kind.
    index: u32,
    addr: usize,
    nbytes: usize,
    /// The object passed to `Region.external`/`from_buffer` (e.g. a tensor), else None.
    obj: Option<Py<PyAny>>,
}

/// A job handed to a Python engine's `submit`.
#[pyclass(name = "CopyJob", module = "hetero_kv", frozen, get_all)]
struct PyCopyJob {
    src: Vec<Py<RegionInfo>>,
    dst: Vec<Py<RegionInfo>>,
    /// `(src_index, src_offset, dst_index, dst_offset, nbytes)` tuples; destinations never overlap.
    spans: Vec<SpanTuple>,
    /// Source representation.
    representation: &'static str,
    /// Destination representation (differs only on converting routes).
    dst_representation: &'static str,
    direction: &'static str,
    /// Source bytes of the job: what `wait` must return.
    nbytes: u64,
}

/// A `CopyEngine` implemented by a Python object with `name`, `capabilities`,
/// `submit(job) -> ticket`, `poll(ticket) -> bool`, `wait(ticket) -> int` and `quiesce()`.
/// Capabilities are `(src, dst, repr | None)` identity routes (None = every representation) or
/// `(src, dst, src_repr, dst_repr)` conversions; `src`/`dst` are device patterns such as
/// "cuda" (any index), "cuda:1" or "host".
struct PyEngine {
    obj: Py<PyAny>,
    name: String,
    caps: Capabilities,
}

impl PyEngine {
    fn new(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        for m in ["submit", "poll", "wait", "quiesce"] {
            if !obj.hasattr(m)? {
                return Err(PyValueError::new_err(format!("engine lacks {m}()")));
            }
        }
        let name: String = obj
            .getattr("name")
            .and_then(|n| n.extract())
            .unwrap_or_else(|_| "python".into());
        let mut caps = Capabilities::new();
        for item in obj.getattr("capabilities")?.try_iter()? {
            let item = item?;
            if let Ok((s, d, a, b)) = item.extract::<(String, String, String, String)>() {
                let (s, d): (DevicePattern, DevicePattern) = (parse(&s)?, parse(&d)?);
                caps = caps.allow_conversion(s, d, parse(&a)?, parse(&b)?);
                continue;
            }
            let (s, d, r): (String, String, Option<String>) = item.extract()?;
            let reprs = match r {
                Some(r) => vec![parse::<Representation>(&r)?],
                None => Representation::ALL.to_vec(),
            };
            let (s, d): (DevicePattern, DevicePattern) = (parse(&s)?, parse(&d)?);
            caps = caps.allow(s, d, &reprs);
        }
        Ok(PyEngine {
            obj: obj.clone().unbind(),
            name,
            caps,
        })
    }
}

fn py_failure(e: PyErr) -> Error {
    Error::Transfer(Python::attach(|py| e.value(py).to_string()))
}

fn region_info(py: Python<'_>, r: &hetero_kv::memory::RegionRef) -> PyResult<Py<RegionInfo>> {
    let obj = r
        .keepalive()
        .and_then(|k| k.downcast_ref::<PyKeep>())
        .map(|k| k.obj.clone_ref(py));
    Py::new(
        py,
        RegionInfo {
            kind: r.kind().as_str(),
            location: r.location().to_string(),
            index: r.location().index,
            addr: r.addr(),
            nbytes: r.len(),
            obj,
        },
    )
}

struct PyCompletion {
    engine: Py<PyAny>,
    ticket: Py<PyAny>,
}

impl PyCompletion {
    fn call_wait(&self, py: Python<'_>) -> hetero_kv::Result<u64> {
        self.engine
            .call_method1(py, "wait", (self.ticket.clone_ref(py),))
            .and_then(|v| v.extract::<u64>(py))
            .map_err(py_failure)
    }
}

impl Completion for PyCompletion {
    fn poll(&mut self) -> hetero_kv::Result<Option<u64>> {
        Python::attach(|py| {
            let done: bool = self
                .engine
                .call_method1(py, "poll", (self.ticket.clone_ref(py),))
                .and_then(|v| v.extract(py))
                .map_err(py_failure)?;
            if done {
                self.call_wait(py).map(Some)
            } else {
                Ok(None)
            }
        })
    }
    fn wait(&mut self) -> hetero_kv::Result<u64> {
        Python::attach(|py| self.call_wait(py))
    }
}

impl CopyEngine for PyEngine {
    fn name(&self) -> &str {
        &self.name
    }
    fn capabilities(&self) -> &Capabilities {
        &self.caps
    }
    fn submit(&self, job: &CopyJob<'_>) -> hetero_kv::Result<Box<dyn Completion>> {
        Python::attach(|py| {
            let infos = |rs: &[hetero_kv::memory::RegionRef]| {
                rs.iter()
                    .map(|r| region_info(py, r))
                    .collect::<PyResult<Vec<_>>>()
            };
            let pyjob = PyCopyJob {
                src: infos(job.src).map_err(py_failure)?,
                dst: infos(job.dst).map_err(py_failure)?,
                spans: job
                    .spans
                    .iter()
                    .map(|s| (s.src, s.src_offset, s.dst, s.dst_offset, s.len))
                    .collect(),
                representation: job.representation.as_str(),
                dst_representation: job.dst_representation.as_str(),
                direction: job.direction.as_str(),
                nbytes: job.bytes(),
            };
            let ticket = self
                .obj
                .call_method1(py, "submit", (pyjob,))
                .map_err(py_failure)?;
            Ok(Box::new(PyCompletion {
                engine: self.obj.clone_ref(py),
                ticket,
            }) as Box<dyn Completion>)
        })
    }
    fn quiesce(&self) -> hetero_kv::Result<()> {
        Python::attach(|py| {
            let r = self.obj.call_method0(py, "quiesce").map_err(py_failure)?;
            if r.bind(py).is_none() || r.extract::<bool>(py).unwrap_or(false) {
                Ok(())
            } else {
                Err(Error::Transfer(format!(
                    "engine {} declined to quiesce",
                    self.name
                )))
            }
        })
    }
}

fn engine_arc(obj: &Bound<'_, PyAny>) -> PyResult<Arc<dyn CopyEngine>> {
    if let Ok(h) = obj.cast::<PyHostEngine>() {
        return Ok(h.get().engine.clone());
    }
    Ok(Arc::new(PyEngine::new(obj)?))
}

// ---------------------------------------------------------------------------------------------
// L3: layouts, specs, buffers, descriptors, progress, events

/// Physical KV layout.
#[pyclass(name = "Layout", module = "hetero_kv", frozen, eq, hash)]
#[derive(Clone, PartialEq, Eq, Hash)]
struct PyLayout {
    inner: Layout,
}

fn geometry(
    representation: &str,
    layers: u32,
    block_size: u32,
    heads: u32,
    head_dim: u32,
    blocks: u32,
    kv: &str,
) -> PyResult<Geometry> {
    let kv = match kv {
        "kv" | "key_value" => KvKind::KeyValue,
        "latent" | "mla" => KvKind::Latent,
        _ => return Err(PyValueError::new_err("kv must be 'kv' or 'latent'")),
    };
    Ok(Geometry {
        representation: parse(representation)?,
        kv,
        layers,
        blocks,
        block_size,
        heads,
        head_dim,
    })
}

fn split_name(s: Split) -> &'static str {
    match s {
        Split::None => "none",
        Split::Layer => "layer",
        Split::LayerKv => "layer_kv",
    }
}

#[pymethods]
impl PyLayout {
    /// Dense layout from an outer-to-inner axis order, or explicit byte `strides`
    /// (dict with layer/kv/block/token/head) plus `region_bytes`.
    #[new]
    #[pyo3(signature = (representation, layers, block_size, heads, head_dim, *, blocks = 1, kv = "kv",
                        order = None, split = "none", strides = None, region_bytes = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        representation: &str,
        layers: u32,
        block_size: u32,
        heads: u32,
        head_dim: u32,
        blocks: u32,
        kv: &str,
        order: Option<Vec<String>>,
        split: &str,
        strides: Option<BTreeMap<String, u64>>,
        region_bytes: Option<u64>,
    ) -> PyResult<Self> {
        let g = geometry(
            representation,
            layers,
            block_size,
            heads,
            head_dim,
            blocks,
            kv,
        )?;
        let split = match split {
            "none" => Split::None,
            "layer" => Split::Layer,
            "layer_kv" => Split::LayerKv,
            _ => {
                return Err(PyValueError::new_err(
                    "split must be none, layer or layer_kv",
                ))
            }
        };
        let inner = match (strides, order) {
            (Some(st), None) => {
                let get = |k: &str| st.get(k).copied().unwrap_or(0);
                let strides = Strides {
                    layer: get("layer"),
                    kv: get("kv"),
                    block: get("block"),
                    token: get("token"),
                    head: get("head"),
                };
                let rb = region_bytes
                    .ok_or_else(|| PyValueError::new_err("strides need region_bytes"))?;
                Layout::strided(g, split, strides, rb).py()?
            }
            (None, order) => {
                let order = order.unwrap_or_else(|| {
                    ["layer", "kv", "block", "token", "head"]
                        .map(String::from)
                        .to_vec()
                });
                let axes: Vec<Axis> = order
                    .iter()
                    .map(|a| Axis::parse(a))
                    .collect::<Result<_, _>>()
                    .py()?;
                let axes: [Axis; 5] = axes
                    .try_into()
                    .map_err(|_| PyValueError::new_err("order must list five axes"))?;
                Layout::contiguous(g, axes, split).py()?
            }
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err("give order or strides, not both"))
            }
        };
        Ok(PyLayout { inner })
    }

    /// Named layout: "layer_separate", "block_major", "layer_major" or "bhsd".
    #[staticmethod]
    #[pyo3(signature = (name, representation, layers, block_size, heads, head_dim, *, blocks = 1, kv = "kv"))]
    #[allow(clippy::too_many_arguments)]
    fn preset(
        name: &str,
        representation: &str,
        layers: u32,
        block_size: u32,
        heads: u32,
        head_dim: u32,
        blocks: u32,
        kv: &str,
    ) -> PyResult<Self> {
        let g = geometry(
            representation,
            layers,
            block_size,
            heads,
            head_dim,
            blocks,
            kv,
        )?;
        let inner = match name {
            "layer_separate" => Layout::layer_separate(g),
            "block_major" => Layout::block_major(g),
            "layer_major" => Layout::layer_major(g),
            "bhsd" => Layout::bhsd(g),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown layout preset {name:?}"
                )))
            }
        }
        .py()?;
        Ok(PyLayout { inner })
    }

    #[getter]
    fn version(&self) -> u32 {
        self.inner.version()
    }
    #[getter]
    fn representation(&self) -> &'static str {
        self.inner.geometry().representation.as_str()
    }
    #[getter]
    fn kv(&self) -> &'static str {
        match self.inner.geometry().kv {
            KvKind::KeyValue => "kv",
            KvKind::Latent => "latent",
        }
    }
    #[getter]
    fn layers(&self) -> u32 {
        self.inner.geometry().layers
    }
    #[getter]
    fn blocks(&self) -> u32 {
        self.inner.geometry().blocks
    }
    #[getter]
    fn block_size(&self) -> u32 {
        self.inner.geometry().block_size
    }
    #[getter]
    fn heads(&self) -> u32 {
        self.inner.geometry().heads
    }
    #[getter]
    fn head_dim(&self) -> u32 {
        self.inner.geometry().head_dim
    }
    #[getter]
    fn split(&self) -> &'static str {
        split_name(self.inner.split())
    }
    #[getter]
    fn strides(&self) -> BTreeMap<&'static str, u64> {
        let s = self.inner.strides();
        BTreeMap::from([
            ("layer", s.layer),
            ("kv", s.kv),
            ("block", s.block),
            ("token", s.token),
            ("head", s.head),
        ])
    }
    #[getter]
    fn region_bytes(&self) -> u64 {
        self.inner.region_bytes()
    }
    #[getter]
    fn region_count(&self) -> usize {
        self.inner.region_count()
    }
    fn __repr__(&self) -> String {
        let g = self.inner.geometry();
        format!(
            "Layout({}, layers={}, blocks={}, block_size={}, heads={}, head_dim={}, split={}, {} x {} bytes)",
            g.representation,
            g.layers,
            g.blocks,
            g.block_size,
            g.heads,
            g.head_dim,
            split_name(self.inner.split()),
            self.inner.region_count(),
            self.inner.region_bytes()
        )
    }
}

/// Model identity plus physical layout.
#[pyclass(name = "KVSpec", module = "hetero_kv", frozen, eq, hash)]
#[derive(Clone, PartialEq, Eq, Hash)]
struct PyKvSpec {
    inner: KvSpec,
}

#[pymethods]
impl PyKvSpec {
    #[new]
    fn new(model_id: String, revision: String, layout: PyLayout) -> PyResult<Self> {
        Ok(PyKvSpec {
            inner: KvSpec::new(model_id, revision, layout.inner).py()?,
        })
    }
    #[getter]
    fn model_id(&self) -> &str {
        &self.inner.model_id
    }
    #[getter]
    fn revision(&self) -> &str {
        &self.inner.revision
    }
    #[getter]
    fn layout(&self) -> PyLayout {
        PyLayout {
            inner: self.inner.layout.clone(),
        }
    }
    #[getter]
    fn representation(&self) -> &'static str {
        self.inner.representation().as_str()
    }
    /// Raise InvalidError unless rows of both specs mean the same thing.
    fn check_compatible(&self, other: &PyKvSpec) -> PyResult<()> {
        self.inner.compatible(&other.inner).py()
    }
    fn __repr__(&self) -> String {
        format!(
            "KVSpec({:?}, {:?}, {})",
            self.inner.model_id,
            self.inner.revision,
            PyLayout {
                inner: self.inner.layout.clone()
            }
            .__repr__()
        )
    }
}

/// A KV buffer: spec plus regions (moved in at construction).
#[pyclass(name = "KVBuffer", module = "hetero_kv", frozen)]
struct PyKvBuffer {
    inner: Mutex<Option<KvBuffer>>,
}

impl PyKvBuffer {
    fn share(&self) -> PyResult<KvBufferRef> {
        self.inner
            .lock()
            .unwrap()
            .as_ref()
            .map(KvBuffer::share)
            .ok_or_else(|| to_py(Error::State("KVBuffer was reclaimed".into())))
    }
}

#[pymethods]
impl PyKvBuffer {
    #[new]
    fn new(spec: &PyKvSpec, regions: Vec<Bound<'_, PyRegion>>) -> PyResult<Self> {
        // Move the regions out; on any failure put every one back untouched.
        let mut taken: Vec<Region> = Vec::new();
        let mut failure = None;
        for r in &regions {
            match r.get().take() {
                Ok(region) => taken.push(region),
                Err(e) => {
                    failure = Some(e);
                    break;
                }
            }
        }
        let failure = failure.or_else(|| KvBuffer::validate(&spec.inner, &taken).err().map(to_py));
        if let Some(e) = failure {
            for (r, region) in regions.iter().zip(taken) {
                *r.get().inner.lock().unwrap() = Some(region);
            }
            return Err(e);
        }
        let buf = KvBuffer::new(spec.inner.clone(), taken).py()?;
        Ok(PyKvBuffer {
            inner: Mutex::new(Some(buf)),
        })
    }

    #[getter]
    fn spec(&self) -> PyResult<PyKvSpec> {
        Ok(PyKvSpec {
            inner: self.share()?.spec().clone(),
        })
    }
    #[getter]
    fn kind(&self) -> PyResult<&'static str> {
        Ok(self.share()?.kind().as_str())
    }
    #[getter]
    fn location(&self) -> PyResult<String> {
        Ok(self.share()?.location().to_string())
    }
    /// Ownership phase of each region ("lent", "in_flight", "done", "uncertain").
    fn phases(&self) -> PyResult<Vec<&'static str>> {
        let g = self.inner.lock().unwrap();
        let b = g
            .as_ref()
            .ok_or_else(|| to_py(Error::State("KVBuffer was reclaimed".into())))?;
        Ok(b.regions().iter().map(|r| r.phase().as_str()).collect())
    }
    /// Take the regions back (fails while a session or registration still references them).
    fn reclaim(&self) -> PyResult<Vec<PyRegion>> {
        let mut g = self.inner.lock().unwrap();
        let b = g
            .take()
            .ok_or_else(|| to_py(Error::State("KVBuffer was reclaimed".into())))?;
        match b.reclaim() {
            Ok(regions) => Ok(regions.into_iter().map(PyRegion::wrap).collect()),
            Err(fail) => {
                let (b, e) = (fail.buffer, fail.error);
                *g = Some(b);
                Err(to_py(e))
            }
        }
    }
    fn __repr__(&self) -> String {
        match self.inner.lock().unwrap().as_ref() {
            Some(b) => format!("{b:?}"),
            None => "KVBuffer(reclaimed)".into(),
        }
    }
}

/// A buffer resolved from a descriptor (possibly another agent's).
#[pyclass(name = "BufferRef", module = "hetero_kv", frozen)]
struct PyBufferRef {
    inner: KvBufferRef,
}

#[pymethods]
impl PyBufferRef {
    #[getter]
    fn spec(&self) -> PyKvSpec {
        PyKvSpec {
            inner: self.inner.spec().clone(),
        }
    }
    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind().as_str()
    }
    #[getter]
    fn location(&self) -> String {
        self.inner.location().to_string()
    }
}

fn buffer_ref(obj: &Bound<'_, PyAny>) -> PyResult<KvBufferRef> {
    if let Ok(b) = obj.cast::<PyKvBuffer>() {
        return b.get().share();
    }
    if let Ok(b) = obj.cast::<PyBufferRef>() {
        return Ok(b.get().inner.clone());
    }
    Err(PyValueError::new_err("expected a KVBuffer or BufferRef"))
}

fn block_hash(obj: &Bound<'_, PyAny>) -> PyResult<BlockHash> {
    if let Ok(b) = obj.cast::<PyBytes>() {
        return Ok(BlockHash::Bytes(b.as_bytes().to_vec()));
    }
    Ok(BlockHash::Int(obj.extract()?))
}

fn block_hash_py(py: Python<'_>, h: &BlockHash) -> PyResult<Py<PyAny>> {
    Ok(match h {
        BlockHash::Int(i) => i.into_pyobject(py)?.into_any().unbind(),
        BlockHash::Bytes(b) => PyBytes::new(py, b).into_any().unbind(),
    })
}

/// Serializable description of a registered KV buffer.
#[pyclass(name = "KVDescriptor", module = "hetero_kv", frozen)]
struct PyKvDescriptor {
    inner: KvDescriptor,
}

#[pymethods]
impl PyKvDescriptor {
    /// Versioned wire bytes.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        Ok(PyKvDescriptor {
            inner: KvDescriptor::from_bytes(data).py()?,
        })
    }
    #[getter]
    fn spec(&self) -> PyKvSpec {
        PyKvSpec {
            inner: self.inner.spec.clone(),
        }
    }
    /// Agent (registry) id the regions belong to.
    #[getter]
    fn agent_id(&self) -> Option<u64> {
        self.inner.regions.first().map(|r| r.registry)
    }
    #[getter]
    fn block_hashes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (k, v) in &self.inner.block_hashes {
            d.set_item(k, block_hash_py(py, v)?)?;
        }
        Ok(d)
    }
    /// Resolve to the registered buffer (P0: agents in this process only).
    fn resolve(&self) -> PyResult<PyBufferRef> {
        Ok(PyBufferRef {
            inner: self.inner.resolve().py()?,
        })
    }
}

fn to_i64(obj: &Bound<'_, PyAny>, msg: &str) -> PyResult<i64> {
    // Exact int only: bool and float are rejected like Optimus' `type(x) is not int` checks.
    if !obj.is_exact_instance_of::<PyInt>() {
        return Err(to_py(Error::Invalid(msg.into())));
    }
    obj.extract().map_err(|_| to_py(Error::Invalid(msg.into())))
}

/// Validated layer-progress coverage for a reserved slab (drop-in for Optimus `LayerProgress`).
#[pyclass(name = "LayerProgress", module = "hetero_kv", frozen)]
struct PyLayerProgress {
    inner: kv::LayerProgress,
}

#[pymethods]
impl PyLayerProgress {
    #[new]
    fn new(
        layers: &Bound<'_, PyAny>,
        start: &Bound<'_, PyAny>,
        end: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let m = "invalid layer-progress geometry";
        Ok(PyLayerProgress {
            inner: kv::LayerProgress::new(to_i64(layers, m)?, to_i64(start, m)?, to_i64(end, m)?)
                .py()?,
        })
    }
    fn publish(
        &self,
        layer_start: &Bound<'_, PyAny>,
        layer_end: &Bound<'_, PyAny>,
        token_start: &Bound<'_, PyAny>,
        token_end: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if self.inner.finished() {
            return Err(to_py(Error::Invalid("layer progress is closed".into())));
        }
        let m = "invalid layer-progress span";
        let span = (
            to_i64(layer_start, m)?,
            to_i64(layer_end, m)?,
            to_i64(token_start, m)?,
            to_i64(token_end, m)?,
        );
        self.inner.publish(span).py()
    }
    fn drain(&self) -> Vec<(i64, i64, i64, i64)> {
        self.inner.drain()
    }
    fn finish(&self) -> PyResult<()> {
        self.inner.finish().py()
    }
    #[getter]
    fn pending(&self) -> bool {
        self.inner.pending()
    }
    #[getter]
    fn finished(&self) -> bool {
        self.inner.finished()
    }
    #[getter]
    fn layers(&self) -> i64 {
        self.inner.layers()
    }
    #[getter]
    fn start(&self) -> i64 {
        self.inner.start()
    }
    #[getter]
    fn end(&self) -> i64 {
        self.inner.end()
    }
}

// ---------------------------------------------------------------------------------------------
// L4: sessions and agents

fn selection(obj: &Bound<'_, PyAny>, layers_default: u32) -> PyResult<Selection> {
    let get = |name: &str| obj.getattr(name);
    let layers: Option<(u32, u32)> = get("layers")?.extract()?;
    let (l0, l1) = layers.unwrap_or((0, layers_default));
    Ok(Selection {
        layers: l0..l1,
        src_blocks: get("src_blocks")?.extract()?,
        dst_blocks: get("dst_blocks")?.extract()?,
        src_start: get("src_start")?.extract()?,
        dst_start: get("dst_start")?.extract()?,
        tokens: get("tokens")?.extract()?,
    })
}

fn receipt_dict<'py>(py: Python<'py>, r: &Receipt) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("request_id", &r.request_id)?;
    d.set_item("engine", &r.engine)?;
    d.set_item("transfers", r.transfers)?;
    d.set_item("spans", r.spans)?;
    d.set_item("bytes", r.bytes)?;
    Ok(d)
}

/// An explicitly selected route over one engine. Share one Route between many sessions (one
/// per request); it is thread-safe and cloning it is cheap.
#[pyclass(name = "Route", module = "hetero_kv", frozen)]
struct PyRoute {
    inner: Route,
}

fn select_route(
    engine: &Bound<'_, PyAny>,
    src: &str,
    dst: &str,
    representation: &str,
    dst_representation: Option<&str>,
) -> PyResult<Route> {
    let (src, dst): (Location, Location) = (parse(src)?, parse(dst)?);
    let from: Representation = parse(representation)?;
    let to = match dst_representation {
        Some(r) => parse(r)?,
        None => from,
    };
    Route::select_converting(engine_arc(engine)?, src, dst, from, to).py()
}

#[pymethods]
impl PyRoute {
    /// Raises UnsupportedError unless `engine` (a `HostEngine` or Python engine object) declared
    /// `src -> dst` for `representation -> dst_representation` (default: no conversion).
    #[new]
    #[pyo3(signature = (engine, src = "host", dst = "host", representation = "bf16", dst_representation = None))]
    fn new(
        engine: &Bound<'_, PyAny>,
        src: &str,
        dst: &str,
        representation: &str,
        dst_representation: Option<&str>,
    ) -> PyResult<Self> {
        Ok(PyRoute {
            inner: select_route(engine, src, dst, representation, dst_representation)?,
        })
    }
    #[getter]
    fn src(&self) -> String {
        self.inner.src().to_string()
    }
    #[getter]
    fn dst(&self) -> String {
        self.inner.dst().to_string()
    }
    #[getter]
    fn representation(&self) -> &'static str {
        self.inner.representation().as_str()
    }
    #[getter]
    fn dst_representation(&self) -> &'static str {
        self.inner.dst_representation().as_str()
    }
    #[getter]
    fn engine(&self) -> String {
        self.inner.engine_name().to_owned()
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// Transfer lifecycle for one destination lease over an explicitly selected route.
#[pyclass(name = "Session", module = "hetero_kv", frozen)]
struct PySession {
    inner: Arc<sess::Session>,
}

#[pymethods]
impl PySession {
    /// `Session(route)` with a shared `Route` (the pattern for many concurrent requests), or
    /// `Session(engine, src, dst, representation, dst_representation)` as a shorthand that
    /// selects a route first. Raises UnsupportedError for an undeclared route; no fallback.
    #[new]
    #[pyo3(signature = (engine, src = "host", dst = "host", representation = "bf16", dst_representation = None))]
    fn new(
        engine: &Bound<'_, PyAny>,
        src: &str,
        dst: &str,
        representation: &str,
        dst_representation: Option<&str>,
    ) -> PyResult<Self> {
        let route = match engine.cast::<PyRoute>() {
            Ok(r) => r.get().inner.clone(),
            Err(_) => select_route(engine, src, dst, representation, dst_representation)?,
        };
        Ok(PySession {
            inner: Arc::new(sess::Session::new(route)),
        })
    }

    #[getter]
    fn state(&self, py: Python<'_>) -> &'static str {
        let s = self.inner.clone();
        py.detach(move || s.state().as_str())
    }
    #[getter]
    fn request_id(&self, py: Python<'_>) -> Option<String> {
        let s = self.inner.clone();
        py.detach(move || s.request_id())
    }
    #[getter]
    fn poison_reason(&self, py: Python<'_>) -> Option<String> {
        let s = self.inner.clone();
        py.detach(move || s.poison_reason())
    }
    /// Number of buffers retained by a poisoned session.
    #[getter]
    fn retained(&self, py: Python<'_>) -> usize {
        let s = self.inner.clone();
        py.detach(move || s.retained())
    }
    #[getter]
    fn engine(&self) -> String {
        self.inner.route().engine_name().to_owned()
    }

    /// Reserve destination buffer(s): one buffer, or a list (one per KV group).
    fn reserve(&self, py: Python<'_>, request_id: String, dst: &Bound<'_, PyAny>) -> PyResult<()> {
        let bufs = match dst.cast::<PyList>() {
            Ok(list) => list
                .iter()
                .map(|b| buffer_ref(&b))
                .collect::<PyResult<Vec<_>>>()?,
            Err(_) => vec![buffer_ref(dst)?],
        };
        let s = self.inner.clone();
        py.detach(move || s.reserve(&request_id, bufs)).py()
    }

    /// Submit a copy of `selection` (a `Selection` or a list of them) from `src` into
    /// destination group `group`. Returns a transfer id.
    #[pyo3(signature = (request_id, src, selection, *, group = 0, direction = "write"))]
    fn transfer(
        &self,
        py: Python<'_>,
        request_id: String,
        src: &Bound<'_, PyAny>,
        selection: &Bound<'_, PyAny>,
        group: usize,
        direction: &str,
    ) -> PyResult<u64> {
        let src = buffer_ref(src)?;
        let layers = src.spec().layers();
        let sels = match selection.cast::<PyList>() {
            Ok(list) => list
                .iter()
                .map(|s| self::selection(&s, layers))
                .collect::<PyResult<Vec<_>>>()?,
            Err(_) => vec![self::selection(selection, layers)?],
        };
        let direction: Direction = parse(direction)?;
        let s = self.inner.clone();
        py.detach(move || s.transfer(&request_id, group, &src, &sels, direction))
            .map(|t| t.0)
            .py()
    }

    fn poll(&self, py: Python<'_>, request_id: String, transfer: u64) -> PyResult<bool> {
        let s = self.inner.clone();
        py.detach(move || s.poll(&request_id, sess::TransferId(transfer)))
            .py()
    }
    fn wait(&self, py: Python<'_>, request_id: String, transfer: u64) -> PyResult<()> {
        let s = self.inner.clone();
        py.detach(move || s.wait(&request_id, sess::TransferId(transfer)))
            .py()
    }
    fn wait_layer(&self, py: Python<'_>, request_id: String, layer: u32) -> PyResult<()> {
        let s = self.inner.clone();
        py.detach(move || s.wait_layer(&request_id, layer)).py()
    }
    fn ready_layers(&self, py: Python<'_>, request_id: String) -> PyResult<Vec<u32>> {
        let s = self.inner.clone();
        py.detach(move || s.ready_layers(&request_id)).py()
    }
    /// Wait for and verify every transfer; returns the receipt dict.
    fn fence<'py>(&self, py: Python<'py>, request_id: String) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.clone();
        let r = py.detach(move || s.fence(&request_id)).py()?;
        receipt_dict(py, &r)
    }
    /// Run `notify()` (make the destination visible) and admit. Requires a prior fence.
    #[pyo3(signature = (request_id, notify = None))]
    fn admit<'py>(
        &self,
        py: Python<'py>,
        request_id: String,
        notify: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.clone();
        let r = py
            .detach(move || {
                s.admit(&request_id, || match &notify {
                    None => Ok(()),
                    Some(f) => Python::attach(|py| f.call0(py).map(|_| ()).map_err(py_failure)),
                })
            })
            .py()?;
        receipt_dict(py, &r)
    }
    #[pyo3(signature = (request_id, passed = true))]
    fn finish(&self, py: Python<'_>, request_id: String, passed: bool) -> PyResult<()> {
        let s = self.inner.clone();
        py.detach(move || s.finish(&request_id, passed)).py()
    }
    fn cancel(&self, py: Python<'_>, request_id: String) -> PyResult<()> {
        let s = self.inner.clone();
        py.detach(move || s.cancel(&request_id)).py()
    }
    /// Quiesce the engine and return a poisoned session to idle.
    fn recover(&self, py: Python<'_>) -> PyResult<()> {
        let s = self.inner.clone();
        py.detach(move || s.recover()).py()
    }
}

fn json_to_py(py: Python<'_>, v: &serde_json::Value) -> PyResult<Py<PyAny>> {
    let s = serde_json::to_string(v).expect("json values serialize");
    Ok(py.import("json")?.call_method1("loads", (s,))?.unbind())
}

/// A named memory domain: registration, descriptors and KV events.
#[pyclass(name = "Agent", module = "hetero_kv", frozen)]
struct PyAgent {
    inner: sess::Agent,
}

#[pymethods]
impl PyAgent {
    #[new]
    #[pyo3(signature = (name, endpoint = None))]
    fn new(name: String, endpoint: Option<String>) -> Self {
        PyAgent {
            inner: sess::Agent::new(name, endpoint),
        }
    }
    #[getter]
    fn id(&self) -> u64 {
        self.inner.id()
    }
    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }
    /// Versioned handshake metadata bytes.
    fn meta<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.meta().to_bytes())
    }
    /// Register `buffer` and describe it; `block_hashes` maps block id -> int | bytes.
    #[pyo3(signature = (buffer, block_hashes = None))]
    fn export(
        &self,
        buffer: &Bound<'_, PyAny>,
        block_hashes: Option<BTreeMap<u32, Bound<'_, PyAny>>>,
    ) -> PyResult<PyKvDescriptor> {
        let hashes = block_hashes
            .unwrap_or_default()
            .into_iter()
            .map(|(k, v)| Ok((k, block_hash(&v)?)))
            .collect::<PyResult<BTreeMap<_, _>>>()?;
        Ok(PyKvDescriptor {
            inner: self.inner.export(&buffer_ref(buffer)?, hashes).py()?,
        })
    }
    fn release(&self, descriptor: &PyKvDescriptor) -> PyResult<()> {
        self.inner.release(&descriptor.inner).py()
    }
    fn resolve(&self, descriptor: &PyKvDescriptor) -> PyResult<PyBufferRef> {
        Ok(PyBufferRef {
            inner: self.inner.resolve(&descriptor.inner).py()?,
        })
    }
    /// Publish a KV event dict, e.g. `{"type": "BlockStored", "block_hashes": [...], ...}`.
    fn publish(&self, py: Python<'_>, event: &Bound<'_, PyDict>) -> PyResult<()> {
        let s: String = py
            .import("json")?
            .call_method1("dumps", (event,))?
            .extract()?;
        let e: KvEvent = serde_json::from_str(&s)
            .map_err(|e| to_py(Error::Invalid(format!("invalid KV event: {e}"))))?;
        self.inner.events().publish(e);
        Ok(())
    }
    /// Drain events as a batch: vLLM array form `[ts, [[tag, ...], ...], dp_rank]` by default,
    /// or `{"ts", "events", "data_parallel_rank"}` with `vllm=False`. None when empty.
    #[pyo3(signature = (data_parallel_rank = None, vllm = true))]
    fn drain_events(
        &self,
        py: Python<'_>,
        data_parallel_rank: Option<i32>,
        vllm: bool,
    ) -> PyResult<Option<Py<PyAny>>> {
        let Some(batch) = self.inner.events().drain(data_parallel_rank) else {
            return Ok(None);
        };
        let v = if vllm {
            batch.to_vllm()
        } else {
            serde_json::to_value(&batch).expect("batches serialize")
        };
        json_to_py(py, &v).map(Some)
    }
}

/// Decode agent handshake bytes into a dict.
#[pyfunction]
fn decode_agent_meta<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyDict>> {
    let m = AgentMeta::from_bytes(data).py()?;
    let d = PyDict::new(py);
    d.set_item("agent_id", m.agent_id)?;
    d.set_item("name", m.name)?;
    d.set_item("endpoint", m.endpoint)?;
    d.set_item("wire_version", m.wire_version)?;
    Ok(d)
}

/// Canonical name of a representation alias (e.g. "torch.bfloat16" -> "bf16").
#[pyfunction]
fn canonical_representation(name: &str) -> PyResult<&'static str> {
    Ok(parse::<Representation>(name)?.as_str())
}

/// Byte-span plan for inspection: list of `(src, src_offset, dst, dst_offset, nbytes)`.
#[pyfunction]
fn plan(src: &PyKvSpec, dst: &PyKvSpec, selection: &Bound<'_, PyAny>) -> PyResult<Vec<SpanTuple>> {
    let sel = self::selection(selection, src.inner.layers())?;
    let p = kv::plan(&src.inner, &dst.inner, &sel).py()?;
    Ok(p.spans()
        .iter()
        .map(|s| (s.src, s.src_offset, s.dst, s.dst_offset, s.len))
        .collect())
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRegion>()?;
    m.add_class::<PyHostEngine>()?;
    m.add_class::<RegionInfo>()?;
    m.add_class::<PyCopyJob>()?;
    m.add_class::<PyLayout>()?;
    m.add_class::<PyKvSpec>()?;
    m.add_class::<PyKvBuffer>()?;
    m.add_class::<PyBufferRef>()?;
    m.add_class::<PyKvDescriptor>()?;
    m.add_class::<PyLayerProgress>()?;
    m.add_class::<PyRoute>()?;
    m.add_class::<PySession>()?;
    m.add_class::<PyAgent>()?;
    m.add_function(wrap_pyfunction!(decode_agent_meta, m)?)?;
    m.add_function(wrap_pyfunction!(plan, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_representation, m)?)?;
    m.add("WIRE_VERSION", hetero_kv::wire::WIRE_VERSION)?;
    m.add("LAYOUT_VERSION", kv::LAYOUT_VERSION)?;
    m.add("MEMORY_KINDS", MemoryKind::ALL.map(|k| k.as_str()).to_vec())?;
    m.add(
        "REPRESENTATIONS",
        Representation::ALL.map(|r| r.as_str()).to_vec(),
    )?;
    Ok(())
}
