//! `ExactIndex`, `VecDelta`: the exact vector array (section 3). Owned by the front thread
//! and shared with nobody; nothing is persisted.

use std::cmp::Ordering;
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
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

/// A flat array of `dim`-wide rows, one key and one scope per row. Row `i` is
/// `data[i * dim..(i + 1) * dim]`.
pub struct ExactIndex {
    dim: usize,
    data: Vec<f32>,
    keys: Vec<VecKey>,
    scopes: Vec<VecScope>,
    pos: HashMap<VecKey, usize>,
}

/// Best first: higher score, then the smaller key.
// SPEC: 2422: the spec does not order ties; they go to the smaller key (`Doc` before `Mem`,
// then the id), so a search is the same whatever order the rows arrived in.
fn best_first(a: &(VecKey, f32), b: &(VecKey, f32)) -> Ordering {
    b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// The document's list of methods (2413-2419) has `len` and no `is_empty`.
#[allow(clippy::len_without_is_empty)]
impl ExactIndex {
    pub fn new(dim: usize) -> Self {
        ExactIndex { dim, data: Vec::new(), keys: Vec::new(), scopes: Vec::new(), pos: HashMap::new() }
    }

    /// `Upsert` of a key already there replaces it.
    // SPEC: 2422: an item whose vector is not `dim` wide is skipped (it cannot be a row).
    pub fn apply(&mut self, delta: VecDelta) {
        match delta {
            VecDelta::Replace(items) => {
                self.data.clear();
                self.keys.clear();
                self.scopes.clear();
                self.pos.clear();
                self.upsert(items);
            }
            VecDelta::Upsert(items) => self.upsert(items),
            VecDelta::Remove(keys) => {
                for key in keys {
                    self.remove(&key);
                }
            }
        }
    }

    fn upsert(&mut self, items: Vec<VecItem>) {
        for item in items {
            if item.v.len() != self.dim {
                continue;
            }
            match self.pos.get(&item.key) {
                Some(&i) => {
                    self.data[i * self.dim..(i + 1) * self.dim].copy_from_slice(&item.v);
                    self.scopes[i] = item.scope;
                }
                None => {
                    self.pos.insert(item.key, self.keys.len());
                    self.keys.push(item.key);
                    self.scopes.push(item.scope);
                    self.data.extend_from_slice(&item.v);
                }
            }
        }
    }

    fn remove(&mut self, key: &VecKey) {
        let Some(i) = self.pos.remove(key) else { return };
        let last = self.keys.len() - 1;
        if i != last {
            let (head, tail) = self.data.split_at_mut(last * self.dim);
            head[i * self.dim..(i + 1) * self.dim].copy_from_slice(&tail[..self.dim]);
            self.pos.insert(self.keys[last], i);
        }
        self.data.truncate(last * self.dim);
        self.keys.swap_remove(i);
        self.scopes.swap_remove(i);
    }

    /// One pass of dot products over the rows `allow` admits, keeping the best `k`, best first.
    pub fn search(&self, q: &[f32], k: usize, allow: &dyn Fn(&VecScope) -> bool) -> Vec<(VecKey, f32)> {
        if k == 0 || q.len() != self.dim {
            return Vec::new();
        }
        let mut hits: Vec<(VecKey, f32)> = Vec::new();
        for (i, key) in self.keys.iter().enumerate() {
            if allow(&self.scopes[i]) {
                hits.push((*key, dot(q, &self.data[i * self.dim..(i + 1) * self.dim])));
            }
        }
        if hits.len() > k {
            hits.select_nth_unstable_by(k - 1, best_first);
            hits.truncate(k);
        }
        hits.sort_unstable_by(best_first);
        hits
    }

    pub fn get(&self, key: &VecKey) -> Option<&[f32]> {
        self.pos.get(key).map(|&i| &self.data[i * self.dim..(i + 1) * self.dim])
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fixed 64-bit LCG (Knuth's MMIX constants); the seed is fixed, never the clock.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn unit(&mut self) -> f32 {
            (self.next() % 2001) as f32 / 1000.0 - 1.0
        }
    }

    const DIM: usize = 16;

    fn scope_of(n: u64) -> VecScope {
        match n % 4 {
            0 => VecScope::Documents,
            1 => VecScope::Global,
            2 => VecScope::Project("/a".to_string()),
            _ => VecScope::Project("/b".to_string()),
        }
    }

    fn random_items(rng: &mut Lcg, n: usize) -> Vec<VecItem> {
        let mut items: Vec<VecItem> = Vec::new();
        for i in 0..n {
            let key = if rng.next().is_multiple_of(2) { VecKey::Doc(i as i64) } else { VecKey::Mem(i as i64) };
            // One row in ten repeats an earlier vector, so ties occur.
            let v = if i > 0 && rng.next().is_multiple_of(10) {
                items[(rng.next() as usize) % items.len()].v.clone()
            } else {
                (0..DIM).map(|_| rng.unit()).collect()
            };
            items.push(VecItem { key, scope: scope_of(rng.next()), v });
        }
        items
    }

    fn brute(items: &[VecItem], q: &[f32], k: usize, allow: &dyn Fn(&VecScope) -> bool) -> Vec<(VecKey, f32)> {
        let mut all: Vec<(VecKey, f32)> =
            items.iter().filter(|it| allow(&it.scope)).map(|it| (it.key, dot(q, &it.v))).collect();
        all.sort_by(best_first);
        all.truncate(k);
        all
    }

    #[test]
    fn search_equals_brute_force_on_1000_random_vectors() {
        let mut rng = Lcg(0x5eed_0009);
        let items = random_items(&mut rng, 1000);
        let mut idx = ExactIndex::new(DIM);
        idx.apply(VecDelta::Replace(items.clone()));
        assert_eq!(idx.len(), 1000);

        let only_a = |s: &VecScope| matches!(s, VecScope::Project(d) if d == "/a");
        let a_or_global = |s: &VecScope| matches!(s, VecScope::Global) || matches!(s, VecScope::Project(d) if d == "/a");
        let filters: [&dyn Fn(&VecScope) -> bool; 3] = [&|_| true, &only_a, &a_or_global];
        let mut ties = 0;
        for round in 0..40 {
            // Some queries are stored vectors themselves.
            let q: Vec<f32> = if round % 4 == 0 {
                items[(rng.next() as usize) % items.len()].v.clone()
            } else {
                (0..DIM).map(|_| rng.unit()).collect()
            };
            for k in [1, 10, 64, 1000, 5000] {
                for allow in filters {
                    let got = idx.search(&q, k, allow);
                    let want = brute(&items, &q, k, allow);
                    assert_eq!(got.len(), want.len());
                    for (g, w) in got.iter().zip(&want) {
                        assert_eq!(g.0, w.0);
                        assert_eq!(g.1.to_bits(), w.1.to_bits());
                    }
                    ties += got.windows(2).filter(|p| p[0].1 == p[1].1).count();
                }
            }
        }
        assert!(ties > 0, "the data must contain ties for the test to mean anything");
    }

    #[test]
    fn upsert_remove_replace_and_scope() {
        let mut idx = ExactIndex::new(3);
        let item = |key, scope: VecScope, v: [f32; 3]| VecItem { key, scope, v: v.to_vec() };
        idx.apply(VecDelta::Upsert(vec![
            item(VecKey::Doc(1), VecScope::Documents, [1.0, 0.0, 0.0]),
            item(VecKey::Mem(1), VecScope::Project("/a".into()), [0.0, 1.0, 0.0]),
            item(VecKey::Mem(2), VecScope::Project("/b".into()), [0.0, 1.0, 0.0]),
            item(VecKey::Mem(3), VecScope::Global, [0.0, 0.0, 1.0]),
        ]));
        assert_eq!(idx.len(), 4);

        // Upsert of an existing key replaces it; len unchanged.
        idx.apply(VecDelta::Upsert(vec![item(VecKey::Mem(1), VecScope::Project("/a".into()), [0.0, 0.0, 1.0])]));
        assert_eq!(idx.len(), 4);
        assert_eq!(idx.get(&VecKey::Mem(1)), Some(&[0.0, 0.0, 1.0][..]));

        // A vector of the wrong width is not a row.
        idx.apply(VecDelta::Upsert(vec![VecItem { key: VecKey::Mem(9), scope: VecScope::Global, v: vec![1.0, 1.0] }]));
        assert_eq!(idx.len(), 4);
        assert_eq!(idx.get(&VecKey::Mem(9)), None);

        // The scope filter never returns a row of another project.
        let only_a = |s: &VecScope| matches!(s, VecScope::Project(d) if d == "/a");
        for q in [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]] {
            let hits = idx.search(&q, 10, &only_a);
            assert!(hits.iter().all(|(k, _)| *k == VecKey::Mem(1)), "{hits:?}");
        }
        let hits = idx.search(&[0.0, 1.0, 0.0], 10, &|s| !matches!(s, VecScope::Project(d) if d == "/a"));
        assert!(hits.iter().all(|(k, _)| *k != VecKey::Mem(1)));
        assert_eq!(hits[0], (VecKey::Mem(2), 1.0));

        // Remove drops it, and the other rows keep their vectors.
        idx.apply(VecDelta::Remove(vec![VecKey::Doc(1), VecKey::Doc(77)]));
        assert_eq!(idx.len(), 3);
        assert_eq!(idx.get(&VecKey::Doc(1)), None);
        assert!(idx.search(&[1.0, 0.0, 0.0], 10, &|_| true).iter().all(|(k, _)| *k != VecKey::Doc(1)));
        assert_eq!(idx.get(&VecKey::Mem(3)), Some(&[0.0, 0.0, 1.0][..]));
        assert_eq!(idx.get(&VecKey::Mem(2)), Some(&[0.0, 1.0, 0.0][..]));
        assert_eq!(idx.get(&VecKey::Mem(1)), Some(&[0.0, 0.0, 1.0][..]));

        // Replace swaps everything.
        idx.apply(VecDelta::Replace(vec![item(VecKey::Doc(5), VecScope::Documents, [0.0, 1.0, 0.0])]));
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.get(&VecKey::Mem(2)), None);
        assert_eq!(idx.search(&[0.0, 1.0, 0.0], 10, &|_| true), vec![(VecKey::Doc(5), 1.0)]);
        assert!(idx.search(&[0.0, 1.0, 0.0], 0, &|_| true).is_empty());
    }
}
