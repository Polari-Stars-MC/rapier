//! Direction-based convex hulls (k-DOP and fixed-direction hulls).
//!
//! A k-DOP (discrete-orientation polytope) or FDH (fixed-direction hull) is the
//! intersection of a set of slabs, each bounded by two parallel planes whose
//! normals come from a fixed set of directions. These are cheap, tight bounding
//! volumes that are popular for broad-phase acceleration and character collision
//! because their overlap test is a handful of dot products.
//!
//! The core routine here projects a point cloud onto a set of directions to get
//! slab bounds, then intersects the corresponding half-spaces to recover the
//! hull vertices and forwards them to `ColliderBuilder::convex_hull`. It is a
//! general-purpose geometry utility (not tied to the collision pipeline) so it
//! can also be reused by compound baking or user spatial queries.
//!
//! The original algorithm and its `KdopHull` / `FdhHull` shape wrappers are kept
//! as a stable public API so host crates can build direction hulls the same way
//! they built AABB / OBB bounds.

use crate::geometry::ColliderBuilder;
use alloc::vec::Vec;
use parry::math::Vector;

const EPSILON: f64 = 1.0e-9;

/// Discrete-orientation polytope preset selecting the slab normal set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KdopPreset {
    /// 6 slabs: the three axis-aligned normals.
    K6,
    /// 14 slabs: K6 plus the four diagonal (±1,±1,±1) normals.
    K14,
    /// 18 slabs: K14 plus the two (±1,±1,0) diagonals on each axis plane.
    K18,
    /// 26 slabs: K18 plus the remaining four (±1,0,±1)/(0,±1,±1) diagonals.
    K26,
}

#[derive(Clone, Copy)]
struct Slab {
    normal: Vector,
    min: f64,
    max: f64,
}

/// A hull defined by a set of slab directions.
pub trait DirectionHull {
    /// The slab normal directions defining this hull.
    fn directions(&self) -> &[Vector];

    /// Build a convex-hull collider from `points` bounded by this hull's
    /// directions. Returns `None` if the points don't span a 3D volume or the
    /// slab intersection is degenerate.
    fn build(&self, points: &[Vector]) -> Option<ColliderBuilder> {
        build_direction_hull(points, self.directions())
    }
}

/// k-DOP hull: owns its direction set (e.g. from a [`KdopPreset`]).
pub struct KdopHull {
    /// The slab normal directions defining this hull.
    pub directions: Vec<Vector>,
}

impl DirectionHull for KdopHull {
    fn directions(&self) -> &[Vector] {
        &self.directions
    }
}

/// Fixed-direction hull: borrows an externally-owned direction set.
pub struct FdhHull<'a> {
    /// The slab normal directions defining this hull.
    pub directions: &'a [Vector],
}

impl DirectionHull for FdhHull<'_> {
    fn directions(&self) -> &[Vector] {
        self.directions
    }
}

/// The canonical slab-normal set for a k-DOP [`KdopPreset`].
pub fn kdop_directions(preset: KdopPreset) -> Vec<Vector> {
    let mut directions: Vec<Vector> = Vec::with_capacity(26);
    directions.push(Vector::new(1.0, 0.0, 0.0));
    directions.push(Vector::new(0.0, 1.0, 0.0));
    directions.push(Vector::new(0.0, 0.0, 1.0));

    if matches!(preset, KdopPreset::K14 | KdopPreset::K18 | KdopPreset::K26) {
        directions.extend([
            Vector::new(1.0, 1.0, 1.0),
            Vector::new(1.0, 1.0, -1.0),
            Vector::new(1.0, -1.0, 1.0),
            Vector::new(-1.0, 1.0, 1.0),
        ]);
    }

    if matches!(preset, KdopPreset::K18 | KdopPreset::K26) {
        directions.extend([
            Vector::new(1.0, 1.0, 0.0),
            Vector::new(1.0, -1.0, 0.0),
        ]);
    }

    if matches!(preset, KdopPreset::K26) {
        directions.extend([
            Vector::new(1.0, 0.0, 1.0),
            Vector::new(1.0, 0.0, -1.0),
            Vector::new(0.0, 1.0, 1.0),
            Vector::new(0.0, 1.0, -1.0),
        ]);
    }

    directions
        .into_iter()
        .filter_map(normalize_direction)
        .collect()
}

fn normalize_direction(direction: Vector) -> Option<Vector> {
    let len = direction.length();
    (len > EPSILON).then_some(direction / len)
}

fn slabs_from_points(points: &[Vector], directions: &[Vector]) -> Option<Vec<Slab>> {
    let mut slabs = Vec::with_capacity(directions.len());
    for direction in directions {
        let Some(normal) = normalize_direction(*direction) else {
            continue;
        };

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for point in points {
            let projection = normal.dot(*point);
            min = min.min(projection);
            max = max.max(projection);
        }

        if min.is_finite() && max.is_finite() {
            slabs.push(Slab { normal, min, max });
        }
    }

    (slabs.len() >= 3).then_some(slabs)
}

fn solve_planes(a: Vector, da: f64, b: Vector, db: f64, c: Vector, dc: f64) -> Option<Vector> {
    let cross_bc = b.cross(c);
    let det = a.dot(cross_bc);
    if det.abs() <= EPSILON {
        return None;
    }

    Some((cross_bc * da + c.cross(a) * db + a.cross(b) * dc) / det)
}

fn contains_point(slabs: &[Slab], point: Vector) -> bool {
    slabs.iter().all(|slab| {
        let projection = slab.normal.dot(point);
        projection >= slab.min - 1.0e-7 && projection <= slab.max + 1.0e-7
    })
}

fn push_unique(points: &mut Vec<Vector>, point: Vector) {
    if points
        .iter()
        .any(|existing| (*existing - point).length_squared() <= 1.0e-12)
    {
        return;
    }

    points.push(point);
}

/// Build a convex-hull collider from `points`, bounded by the slab directions in
/// `directions`.
///
/// Returns `None` when `points` has fewer than 4 entries (no 3D volume), or when
/// the slab intersection is degenerate and yields no valid hull vertices.
pub fn build_direction_hull(points: &[Vector], directions: &[Vector]) -> Option<ColliderBuilder> {
    if points.len() < 4 {
        return None;
    }

    let slabs = slabs_from_points(points, directions)?;
    let mut planes: Vec<(Vector, f64)> = Vec::with_capacity(slabs.len() * 2);
    for slab in &slabs {
        planes.push((slab.normal, slab.max));
        planes.push((-slab.normal, -slab.min));
    }

    let mut vertices: Vec<Vector> = Vec::new();
    for i in 0..planes.len() {
        for j in (i + 1)..planes.len() {
            for k in (j + 1)..planes.len() {
                let Some(point) = solve_planes(
                    planes[i].0,
                    planes[i].1,
                    planes[j].0,
                    planes[j].1,
                    planes[k].0,
                    planes[k].1,
                ) else {
                    continue;
                };

                if contains_point(&slabs, point) {
                    push_unique(&mut vertices, point);
                }
            }
        }
    }

    ColliderBuilder::convex_hull(vertices.as_slice())
}
