//! `ExactIndex`, `VecDelta`: the exact vector array. Commit 9 writes this file.

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum VecKey {
    /// `documents.id`.
    Doc(i64),
    /// `memory_ext.vid`.
    Mem(i64),
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum VecScope {
    Documents,
    Global,
    /// `scope_dir`.
    Project(String),
}

#[derive(Clone, Debug)]
pub struct VecItem {
    pub key: VecKey,
    pub scope: VecScope,
    pub v: Vec<f32>,
}

#[derive(Clone, Debug)]
pub enum VecDelta {
    Replace(Vec<VecItem>),
    Upsert(Vec<VecItem>),
    Remove(Vec<VecKey>),
}

// SPEC: 2342-2348 give `ExactIndex`'s methods and not its fields. The stub keeps the
// dimension and holds nothing.
pub struct ExactIndex {
    // Nothing reads it until commit 9.
    #[allow(dead_code)]
    dim: usize,
}

// The document's list of methods (2342-2348) has `len` and no `is_empty`.
#[allow(clippy::len_without_is_empty)]
impl ExactIndex {
    pub fn new(dim: usize) -> Self {
        ExactIndex { dim }
    }

    /// `Upsert` of a key already there replaces it.
    pub fn apply(&mut self, delta: VecDelta) {
        let _ = delta;
    }

    pub fn search(&self, q: &[f32], k: usize, allow: &dyn Fn(&VecScope) -> bool) -> Vec<(VecKey, f32)> {
        let _ = (q, k, allow);
        Vec::new()
    }

    pub fn get(&self, key: &VecKey) -> Option<&[f32]> {
        let _ = key;
        None
    }

    pub fn len(&self) -> usize {
        0
    }
}
