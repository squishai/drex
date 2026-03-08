use pyo3::prelude::*;

mod error;
mod storage;
mod cache;
mod prefetch;
mod bindings;

use bindings::{
    storage_py::PySnapshotStore,
    storage_py::PySnapshotMeta,
    cache_py::PyMemoryTierManager,
    prefetch_py::PyPrefetchEngine,
};

/// drex._sys — Rust systems layer for the Drex LLM architecture.
///
/// Provides high-performance disk I/O, memory tier management, H2O eviction,
/// and InfiniGen-style speculative prefetch for the L3 disk cache tier.
#[pymodule]
fn _sys(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySnapshotStore>()?;
    m.add_class::<PySnapshotMeta>()?;
    m.add_class::<PyMemoryTierManager>()?;
    m.add_class::<PyPrefetchEngine>()?;
    Ok(())
}
