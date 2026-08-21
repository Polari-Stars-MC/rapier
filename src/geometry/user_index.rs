//! User-space axis-aligned bounding-box spatial index.
//!
//! This is a general-purpose, dependency-light R-tree over 3D AABBs that the
//! host application can populate with arbitrary `u64` entry ids and query by
//! AABB intersection. It is intentionally independent of the collision-detection
//! broad-phase: the engine's internal [`crate::geometry::BroadPhaseBvh`] serves
//! collider pairs, whereas this index is for user-managed spatial lookups
//! (e.g. game-world entity buckets, region queries) where the caller owns the
//! ids and the AABBs.
//!
//! The implementation mirrors a classic bulk-loaded R-tree: entries are kept in a
//! flat `Vec` and a balanced tree is (lazily) rebuilt whenever the set is
//! mutated. Queries prune whole subtrees whose bounds don't intersect the query
//! AABB.

use crate::geometry::Aabb;
use crate::parry::bounding_volume::BoundingVolume;
use alloc::vec::Vec;
use parry::math::Vector;

const MAX_CHILDREN: usize = 8;

/// A single indexed AABB entry.
#[derive(Clone, Copy, Debug)]
struct Entry {
    id: u64,
    bounds: Aabb,
}

#[derive(Clone, Debug)]
enum NodeKind {
    Leaf(Vec<Entry>),
    Branch(Vec<Node>),
}

#[derive(Clone, Debug)]
struct Node {
    bounds: Aabb,
    kind: NodeKind,
}

/// A user-managed spatial index over 3D AABBs keyed by `u64` entry ids.
///
/// Insertions and removals only mark the index dirty; the underlying tree is
/// rebuilt lazily on the next query (or on an explicit [`Self::rebuild`]). This
/// keeps burst mutations cheap at the cost of one rebuild per query burst.
#[derive(Clone, Debug)]
pub struct GenericAabbIndex {
    entries: Vec<Entry>,
    root: Option<Node>,
    dirty: bool,
}

impl GenericAabbIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            root: None,
            dirty: false,
        }
    }

    /// Remove every entry from the index.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.root = None;
        self.dirty = false;
    }

    /// Insert or overwrite the bounds of `id`.
    ///
    /// Returns `false` (and makes no change) if `id == 0`, which is reserved as
    /// a sentinel. Capacity limits are the caller's responsibility; this index
    /// itself is unbounded.
    pub fn insert(&mut self, id: u64, bounds: Aabb) -> bool {
        if id == 0 {
            return false;
        }

        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.id == id) {
            entry.bounds = bounds;
        } else {
            self.entries.push(Entry { id, bounds });
        }
        self.dirty = true;
        true
    }

    /// Remove `id` from the index. Returns `true` if it was present.
    pub fn remove(&mut self, id: u64) -> bool {
        let Some(index) = self.entries.iter().position(|entry| entry.id == id) else {
            return false;
        };
        self.entries.swap_remove(index);
        self.dirty = true;
        true
    }

    /// Number of entries currently stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the index holds no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether the index currently contains an entry with the given `id`.
    pub fn contains(&self, id: u64) -> bool {
        self.entries.iter().any(|entry| entry.id == id)
    }

    /// Force an immediate rebuild of the tree structure.
    pub fn rebuild(&mut self) {
        self.rebuild_if_needed();
    }

    fn rebuild_if_needed(&mut self) {
        if !self.dirty {
            return;
        }
        self.root = build_node(&mut self.entries);
        self.dirty = false;
    }

    /// Count the entries whose bounds intersect `bounds`.
    pub fn query_count(&mut self, bounds: Aabb) -> u32 {
        self.rebuild_if_needed();
        let Some(root) = &self.root else {
            return 0;
        };
        count_node(root, bounds)
    }

    /// Write the ids of entries whose bounds intersect `bounds` into `out_ids`.
    ///
    /// Returns the number of ids written (capped at `out_ids.len()`).
    pub fn query(&mut self, bounds: Aabb, out_ids: &mut [u64]) -> u32 {
        self.rebuild_if_needed();
        let Some(root) = &self.root else {
            return 0;
        };
        let mut written = 0usize;
        query_node(root, bounds, out_ids, &mut written);
        written as u32
    }
}

impl Default for GenericAabbIndex {
    fn default() -> Self {
        Self::new()
    }
}

fn aabb_union(a: Aabb, b: Aabb) -> Aabb {
    Aabb::new(
        Vector::new(
            a.mins.x.min(b.mins.x),
            a.mins.y.min(b.mins.y),
            a.mins.z.min(b.mins.z),
        ),
        Vector::new(
            a.maxs.x.max(b.maxs.x),
            a.maxs.y.max(b.maxs.y),
            a.maxs.z.max(b.maxs.z),
        ),
    )
}

fn aabb_center_axis(b: &Aabb, axis: usize) -> f64 {
    match axis {
        0 => (b.mins.x + b.maxs.x) * 0.5,
        1 => (b.mins.y + b.maxs.y) * 0.5,
        _ => (b.mins.z + b.maxs.z) * 0.5,
    }
}

fn aabb_extent_axis(b: &Aabb, axis: usize) -> f64 {
    match axis {
        0 => b.maxs.x - b.mins.x,
        1 => b.maxs.y - b.mins.y,
        _ => b.maxs.z - b.mins.z,
    }
}

fn entries_bounds(entries: &[Entry]) -> Option<Aabb> {
    let mut iter = entries.iter();
    let first = iter.next()?.bounds;
    Some(iter.fold(first, |acc, entry| aabb_union(acc, entry.bounds)))
}

fn nodes_bounds(nodes: &[Node]) -> Option<Aabb> {
    let mut iter = nodes.iter();
    let first = iter.next()?.bounds;
    Some(iter.fold(first, |acc, node| aabb_union(acc, node.bounds)))
}

fn longest_axis(bounds: &Aabb) -> usize {
    let x = aabb_extent_axis(bounds, 0);
    let y = aabb_extent_axis(bounds, 1);
    let z = aabb_extent_axis(bounds, 2);
    if x >= y && x >= z {
        0
    } else if y >= z {
        1
    } else {
        2
    }
}

fn build_node(entries: &mut [Entry]) -> Option<Node> {
    let bounds = entries_bounds(entries)?;
    if entries.len() <= MAX_CHILDREN {
        return Some(Node {
            bounds,
            kind: NodeKind::Leaf(entries.to_vec()),
        });
    }

    let axis = longest_axis(&bounds);
    entries.sort_unstable_by(|a, b| {
        aabb_center_axis(&a.bounds, axis)
            .total_cmp(&aabb_center_axis(&b.bounds, axis))
            .then_with(|| a.id.cmp(&b.id))
    });

    let child_count = entries.len().div_ceil(MAX_CHILDREN);
    let mut children = Vec::with_capacity(child_count);
    for chunk in entries.chunks_mut(MAX_CHILDREN) {
        if let Some(child) = build_node(chunk) {
            children.push(child);
        }
    }

    let bounds = nodes_bounds(&children)?;
    Some(Node {
        bounds,
        kind: NodeKind::Branch(children),
    })
}

fn count_node(node: &Node, bounds: Aabb) -> u32 {
    if !node.bounds.intersects(&bounds) {
        return 0;
    }

    match &node.kind {
        NodeKind::Leaf(entries) => entries
            .iter()
            .filter(|entry| entry.bounds.intersects(&bounds))
            .count() as u32,
        NodeKind::Branch(children) => children
            .iter()
            .map(|child| count_node(child, bounds))
            .sum::<u32>(),
    }
}

fn query_node(node: &Node, bounds: Aabb, out_ids: &mut [u64], written: &mut usize) {
    if *written >= out_ids.len() || !node.bounds.intersects(&bounds) {
        return;
    }

    match &node.kind {
        NodeKind::Leaf(entries) => {
            for entry in entries.iter() {
                if *written >= out_ids.len() {
                    return;
                }
                if entry.bounds.intersects(&bounds) {
                    out_ids[*written] = entry.id;
                    *written += 1;
                }
            }
        }
        NodeKind::Branch(children) => {
            for child in children {
                query_node(child, bounds, out_ids, written);
            }
        }
    }
}
