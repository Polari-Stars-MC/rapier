//! Regression tests for upstream issue #993 ("Voxel-ball bugs — ghost collisions and
//! non-collisions").
//!
//! A ball fired at the edge/corner region of a block of solid voxels used to tunnel into
//! the block's interior: the ball-vs-voxels contact manifold is generated **per voxel**,
//! and each voxel projects the ball's center onto its own "pseudo-cube" independently. On
//! an octant whose feature is `INTERIOR` (the voxel face/edge is buried inside the solid
//! block) the projection returns `None`, so *no* contact is produced there. When the ball
//! straddles the boundary between an exposed voxel and its buried neighbour, the manifolds
//! that do get produced can point the wrong way (or vanish entirely for a step), letting
//! the ball slip past the surface and rattle around inside the block.
//!
//! The test below mirrors the TypeScript reproduction from the issue: a 3×3×3 block of unit
//! voxels spanning grid coordinates 5..=7, and a ball of radius 0.25 launched at its top
//! edge with the exact position/velocity from the report. After stepping, the ball must stay
//! *outside* the solid block — never inside its interior.

use crate::alloc_prelude::*;
use crate::dynamics::{
    CCDSolver, ImpulseJointSet, IntegrationParameters, IslandManager, MultibodyJointSet,
    RigidBodyBuilder, RigidBodySet,
};
use crate::geometry::{ColliderBuilder, ColliderSet, DefaultBroadPhase, NarrowPhase};
use crate::math::Vector;
use crate::pipeline::PhysicsPipeline;
use parry::math::IVector;

/// Builds the 3×3×3 block of solid voxels spanning grid coords `5..=7` on each axis,
/// exactly like the issue's `voxelData`.
fn issue_993_voxel_block() -> Vec<IVector> {
    let mut voxels = Vec::new();
    for y in 5..=7 {
        for z in 5..=7 {
            for x in 5..=7 {
                voxels.push(IVector::new(x, y, z));
            }
        }
    }
    voxels
}

/// The solid block occupies world AABB [5, 8]³ (unit voxels, grid coord `i` spans
/// `[i, i+1]`). Returns true when `p` is strictly inside that box, inset by the ball
/// radius so "inside" means the ball's center got past the surface rather than merely
/// touching it.
fn is_inside_block(p: Vector, inset: f64) -> bool {
    let lo = 5.0 + inset;
    let hi = 8.0 - inset;
    p.x > lo && p.x < hi && p.y > lo && p.y < hi && p.z > lo && p.z < hi
}

/// Issue #993: a ball shot at the top edge of a solid voxel block must bounce off it,
/// not tunnel into the block's interior.
#[test]
fn ball_does_not_tunnel_into_voxel_block() {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();

    // Static voxel terrain (unit voxels).
    let voxels = issue_993_voxel_block();
    colliders.insert(
        ColliderBuilder::voxels(Vector::new(1.0, 1.0, 1.0), &voxels)
            .friction(0.8)
            .restitution(0.0),
    );

    // Projectile: exact initial state from the issue's reproduction.
    let projectile_radius = 0.25;
    let rb = RigidBodyBuilder::dynamic()
        .translation(Vector::new(
            5.068_835_807_168_076,
            10.666_715_670_475_918,
            10.803_566_521_990_922,
        ))
        .linvel(Vector::new(
            5.966_327_745_035_736,
            -12.776_961_047_623_74,
            -14.182_813_529_984_893,
        ))
        .ccd_enabled(true)
        .build();
    let ball = bodies.insert(rb);
    let ball_collider = colliders.insert_with_parent(
        ColliderBuilder::ball(projectile_radius)
            .density(1.0)
            .friction(0.8)
            .restitution(0.45),
        ball,
        &mut bodies,
    );

    let mut pipeline = PhysicsPipeline::new();
    let mut islands = IslandManager::new();
    let mut broad_phase = DefaultBroadPhase::new();
    let mut narrow_phase = NarrowPhase::new();
    let mut impulse_joints = ImpulseJointSet::new();
    let mut multibody_joints = MultibodyJointSet::new();
    let mut ccd = CCDSolver::new();
    let params = IntegrationParameters {
        dt: 1.0 / 60.0,
        ..IntegrationParameters::default()
    };
    let gravity = Vector::new(0.0, -10.0, 0.0);

    // The issue reports the ball fully inside the voxels after 15-20 ticks.
    let start = bodies[ball].translation();
    let mut contact_ticks = 0usize;
    for tick in 0..40 {
        pipeline.step(
            gravity,
            &params,
            &mut islands,
            &mut broad_phase,
            &mut narrow_phase,
            &mut bodies,
            &mut colliders,
            &mut impulse_joints,
            &mut multibody_joints,
            &mut ccd,
            &(),
            &(),
        );

        let p = bodies[ball].translation();
        assert!(
            !is_inside_block(p, projectile_radius),
            "issue #993: ball tunneled into the solid voxel block at tick {tick}: \
             position=({:.4}, {:.4}, {:.4})",
            p.x,
            p.y,
            p.z
        );
        if narrow_phase
            .contact_pairs_with(ball_collider)
            .any(|pair| pair.has_any_active_contact())
        {
            contact_ticks += 1;
        }
    }

    // Sanity: the ball must actually collide with the voxel block, otherwise the assertion
    // above would pass trivially (a ball that never arrives cannot tunnel).
    assert!(
        contact_ticks > 0,
        "test is vacuous: ball never contacted the voxel block; started at ({:.3}, {:.3}, {:.3})",
        start.x,
        start.y,
        start.z
    );
}
