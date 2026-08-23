//! Phase C: baked compound collision.
//!
//! Box3D-style "uber shape" pre-cooking for large static environments. A `Compound`
//! (parry's `SharedShape::compound`) already builds and stores a `Bvh` for its
//! sub-shapes; under parry's `serde` feature that BVH is serde-derivable, so the
//! bake format is just a serialized, ready-to-use shape plus the mass properties
//! computed at bake time. Reloading a baked compound constructs a `ColliderBuilder`
//! whose shape/geometry BVH is reused verbatim — no BVH rebuild on load.
//!
//! This is gated behind `feature = "serde-serialize"` because the bake format is a
//! serde payload (mirrors the rest of Rapier's serialization, which is opt-in).

#![cfg(feature = "serde-serialize")]

use crate::alloc_prelude::*;
use crate::geometry::SharedShape;
use crate::geometry::collider::ColliderBuilder;
use crate::geometry::collider_components::ColliderMassProps;
use serde::{Deserialize, Serialize};

/// A serializable snapshot of a baked compound collider.
///
/// The `shape` field carries parry's `Compound` (including its pre-built BVH),
/// so a deserialize restores the fully-built query structure without rebuilding
/// the BVH. Mass properties are frozen at bake time so `from_baked_compound`
/// restores identical dynamics to the original builder.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BakedCompound {
    /// The cooked shape (a `Compound` with its BVH embedded).
    shape: SharedShape,
    /// Mass properties captured when the compound was baked.
    mass_properties: ColliderMassProps,
    /// Whether the baked collider is a sensor.
    is_sensor: bool,
}

impl BakedCompound {
    /// Serializes this baked compound to compact bytes (bincode little-endian).
    ///
    /// The payload includes the pre-built BVH, so loading the bytes is much cheaper
    /// than reconstructing the compound from its raw sub-shapes. The output is
    /// suitable for on-disk caching or memory-mapped streaming loaders.
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }

    /// Deserializes a baked compound previously produced by [`Self::to_bytes`].
    ///
    /// Reuses the embedded BVH; no BVH construction is performed.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(bytes)
    }
}

impl ColliderBuilder {
    /// Phase C: bakes this builder's shape into a [`BakedCompound`].
    ///
    /// Any `SharedShape` can be baked, but the primary use case is a `Compound`
    /// built from many sub-shapes — the bake captures its pre-built BVH so loading
    /// is BVH-rebuild-free. The mass properties are captured from the builder's
    /// current `mass_properties` setting (density/mass/units) so the baked form is
    /// dynamics-equivalent to the original.
    pub fn bake_compound(&self) -> BakedCompound {
        BakedCompound {
            shape: self.shape.clone(),
            mass_properties: self.mass_properties.clone(),
            is_sensor: self.is_sensor,
        }
    }

    /// Phase C: rebuilds a [`ColliderBuilder`] from a [`BakedCompound`].
    ///
    /// Restores the cooked shape (with its pre-built BVH) and the bake-time mass
    /// properties and sensor flag. Material, collision groups, hooks, events,
    /// position and user-data are left at their builder defaults and should be set
    /// by the caller as needed — baking captures geometry and dynamics only.
    pub fn from_baked_compound(baked: BakedCompound) -> Self {
        let mut builder = ColliderBuilder::new(baked.shape);
        builder.set_mass_properties_spec(baked.mass_properties);
        builder.is_sensor = baked.is_sensor;
        builder
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::geometry::ColliderMassProps;
    use crate::geometry::{ColliderBuilder, SharedShape};
    use crate::math::Pose;

    /// Phase C: a baked compound must round-trip through bytes and rebuild a
    /// geometry-equivalent collider (same AABB, same mass-properties spec) without
    /// needing to reconstruct the BVH from its raw sub-shapes.
    #[test]
    fn bake_compound_round_trips_through_bytes() {
        let sub_shapes = vec![
            (Pose::translation(0.0, 0.0, 0.0), SharedShape::ball(0.5)),
            (Pose::translation(2.0, 0.0, 0.0), SharedShape::ball(0.5)),
            (
                Pose::translation(4.0, 1.0, 0.0),
                SharedShape::cuboid(0.3, 0.3, 0.3),
            ),
        ];
        let original_builder = ColliderBuilder::compound(sub_shapes).density(2.0);
        let original_aabb = original_builder.clone().build().compute_aabb();

        // Bake the builder (captures shape + mass-props spec) and serialize.
        let baked = original_builder.bake_compound();
        let bytes = baked.to_bytes().expect("bake -> bytes");

        // Deserialize + rebuild.
        let restored = BakedCompound::from_bytes(&bytes).expect("bytes -> baked");
        let rebuilt_builder = ColliderBuilder::from_baked_compound(restored);
        let rebuilt_aabb = rebuilt_builder.clone().build().compute_aabb();

        // Geometry round-trips exactly (the baked BVH is reused, not rebuilt).
        let dmin = (original_aabb.mins.x - rebuilt_aabb.mins.x).abs()
            + (original_aabb.mins.y - rebuilt_aabb.mins.y).abs()
            + (original_aabb.mins.z - rebuilt_aabb.mins.z).abs();
        let dmax = (original_aabb.maxs.x - rebuilt_aabb.maxs.x).abs()
            + (original_aabb.maxs.y - rebuilt_aabb.maxs.y).abs()
            + (original_aabb.maxs.z - rebuilt_aabb.maxs.z).abs();
        assert!(dmin < 1e-6, "baked AABB mins diverged");
        assert!(dmax < 1e-6, "baked AABB maxs diverged");

        // Mass-properties spec preserved.
        assert_eq!(
            rebuilt_builder.mass_properties,
            ColliderMassProps::Density(2.0)
        );
    }
}
