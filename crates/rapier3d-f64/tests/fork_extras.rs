//! The fork layer fused into the official soft body: wind and body damping.
//!
//! These pin the solver-side wiring of `dynamics::soft_body::fork_extras`: wind is an
//! acceleration on every free particle, and the fork damping is a per-step velocity scale
//! applied after the solve. Both are inert by default, so upstream's own soft-body tests
//! stay the reference for the untouched path.

use rapier3d_f64::prelude::*;

/// Two particles linked by one edge: the smallest body the XPBD path simulates.
fn two_particle_body() -> (RigidBodySet, ColliderSet, SoftBodySet, SoftBodyHandle) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut soft_bodies = SoftBodySet::new();
    let builder =
        SoftBodyBuilder::new(vec![Vector::new(0.0, 0.0, 0.0), Vector::new(0.0, 1.0, 0.0)])
            .edges(vec![[0, 1]])
            .no_surface_collider();
    let handle = soft_bodies.insert(builder, &mut bodies, &mut colliders);
    (bodies, colliders, soft_bodies, handle)
}

fn mean_x(sb: &SoftBody) -> Real {
    let n = sb.num_particles() as Real;
    sb.particle_positions().map(|p| p.x).sum::<Real>() / n
}

fn mean_speed(sb: &SoftBody) -> Real {
    let n = sb.num_particles().max(1) as Real;
    sb.particle_velocities().map(|v| v.length()).sum::<Real>() / n
}

/// Steps `soft_bodies` for `steps` frames of `dt` with zero gravity, ignoring contacts.
#[allow(clippy::too_many_arguments)]
fn step(
    soft_bodies: &mut SoftBodySet,
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    pipeline: &mut PhysicsPipeline,
    islands: &mut IslandManager,
    broad_phase: &mut DefaultBroadPhase,
    narrow_phase: &mut NarrowPhase,
    ccd: &mut CCDSolver,
    impulse_joints: &mut ImpulseJointSet,
    multibody_joints: &mut MultibodyJointSet,
    params: &IntegrationParameters,
    steps: usize,
) {
    for _ in 0..steps {
        pipeline.step(
            Vector::ZERO,
            params,
            islands,
            broad_phase,
            narrow_phase,
            bodies,
            colliders,
            impulse_joints,
            multibody_joints,
            soft_bodies,
            ccd,
            &(),
            &(),
        );
    }
}

#[test]
fn wind_accelerates_every_free_particle() {
    let (mut bodies, mut colliders, mut soft_bodies, handle) = two_particle_body();
    let mut pipeline = PhysicsPipeline::new();
    let mut islands = IslandManager::new();
    let mut broad_phase = DefaultBroadPhase::new();
    let mut narrow_phase = NarrowPhase::new();
    let mut ccd = CCDSolver::new();
    let mut impulse_joints = ImpulseJointSet::new();
    let mut multibody_joints = MultibodyJointSet::new();
    let params = IntegrationParameters::default();

    // 1 m/s² along +x, no drag.
    soft_bodies[handle].set_wind_field(Some(Wind {
        accel: Vector::new(1.0, 0.0, 0.0),
        drag: 0.0,
    }));

    step(
        &mut soft_bodies,
        &mut bodies,
        &mut colliders,
        &mut pipeline,
        &mut islands,
        &mut broad_phase,
        &mut narrow_phase,
        &mut ccd,
        &mut impulse_joints,
        &mut multibody_joints,
        &params,
        60,
    );

    let dt = params.dt;
    // Semi-implicit Euler: x = a·dt²·Σi = a·dt²·n(n+1)/2 for n = 60 steps of a = 1 m/s².
    let expected = dt * dt * (60.0 * 61.0 / 2.0);
    let x = mean_x(soft_bodies.get(handle).unwrap());
    assert!(
        (x - expected).abs() < 0.02,
        "wind should integrate to {expected}, got {x}"
    );
}

#[test]
fn fork_damping_removes_velocity_every_step() {
    let params = IntegrationParameters::default();
    let mut speeds = Vec::new();

    for damping in [None, Some(0.2)] {
        let (mut bodies, mut colliders, mut soft_bodies, handle) = two_particle_body();
        let mut pipeline = PhysicsPipeline::new();
        let mut islands = IslandManager::new();
        let mut broad_phase = DefaultBroadPhase::new();
        let mut narrow_phase = NarrowPhase::new();
        let mut ccd = CCDSolver::new();
        let mut impulse_joints = ImpulseJointSet::new();
        let mut multibody_joints = MultibodyJointSet::new();

        let sb = &mut soft_bodies[handle];
        sb.set_particle_velocity(0, Vector::new(1.0, 0.0, 0.0));
        sb.set_particle_velocity(1, Vector::new(1.0, 0.0, 0.0));
        if let Some(d) = damping {
            assert!(sb.set_fork_damping(d), "0.2 is a valid damping");
        }

        step(
            &mut soft_bodies,
            &mut bodies,
            &mut colliders,
            &mut pipeline,
            &mut islands,
            &mut broad_phase,
            &mut narrow_phase,
            &mut ccd,
            &mut impulse_joints,
            &mut multibody_joints,
            &params,
            30,
        );
        speeds.push(mean_speed(soft_bodies.get(handle).unwrap()));
    }

    let undamped = speeds[0];
    let damped = speeds[1];
    assert!(
        undamped > 0.9,
        "an unforced body keeps its velocity, got {undamped}"
    );
    // (1 − 0.2)^30 ≈ 1.2e-3: the damped body must be essentially stopped.
    assert!(
        damped < undamped * 0.05,
        "damped speed {damped} should be far below {undamped}"
    );
}

#[test]
fn invalid_fork_parameters_are_rejected() {
    let (_bodies, _colliders, mut soft_bodies, handle) = two_particle_body();
    let sb = &mut soft_bodies[handle];

    assert!(!sb.set_fork_damping(1.0), "damping 1 would freeze the body");
    assert!(!sb.set_fork_damping(-0.1));
    assert!(!sb.set_active_strain(1.5));
    assert!(!sb.set_extra_substeps(0));
    assert!(!sb.set_self_contact(Some(SelfCollisionParams {
        radius: -1.0,
        stiffness: 0.0,
        friction: 0.0,
    })));
    assert_eq!(sb.fork_extras().damping, 0.0, "rejections leave it inert");
}

/// Two particles, the first one pinned: the distance constraint drives the free one.
fn pinned_two_particle_body() -> (RigidBodySet, ColliderSet, SoftBodySet, SoftBodyHandle) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut soft_bodies = SoftBodySet::new();
    let builder =
        SoftBodyBuilder::new(vec![Vector::new(0.0, 0.0, 0.0), Vector::new(0.0, 1.0, 0.0)])
            .edges(vec![[0, 1]])
            .pinned_particles([0])
            .no_surface_collider();
    let handle = soft_bodies.insert(builder, &mut bodies, &mut colliders);
    (bodies, colliders, soft_bodies, handle)
}

fn edge_length(sb: &SoftBody) -> Real {
    (sb.particle_position(1) - sb.particle_position(0)).length()
}

/// Runs a pinned two-particle body for `steps` frames with the given fork tweak.
fn pinned_edge_length(tweak: impl FnOnce(&mut SoftBody), steps: usize) -> Real {
    let (mut bodies, mut colliders, mut soft_bodies, handle) = pinned_two_particle_body();
    let mut pipeline = PhysicsPipeline::new();
    let mut islands = IslandManager::new();
    let mut broad_phase = DefaultBroadPhase::new();
    let mut narrow_phase = NarrowPhase::new();
    let mut ccd = CCDSolver::new();
    let mut impulse_joints = ImpulseJointSet::new();
    let mut multibody_joints = MultibodyJointSet::new();
    let params = IntegrationParameters::default();

    tweak(&mut soft_bodies[handle]);
    step(
        &mut soft_bodies,
        &mut bodies,
        &mut colliders,
        &mut pipeline,
        &mut islands,
        &mut broad_phase,
        &mut narrow_phase,
        &mut ccd,
        &mut impulse_joints,
        &mut multibody_joints,
        &params,
        steps,
    );
    edge_length(soft_bodies.get(handle).unwrap())
}

#[test]
fn thermal_expansion_lengthens_the_rest_edge() {
    let base = pinned_edge_length(|_| {}, 120);
    let hot = pinned_edge_length(
        |sb| {
            assert!(sb.set_thermal_field(Some(ThermalParams {
                temp: 100.0,
                ambient: 0.0,
                expansion: 0.01,
                stiffness_temp_coeff: 0.0,
            })));
        },
        120,
    );
    // rest 1 m × (1 + 0.01·100) = 2 m.
    assert!(
        (base - 1.0).abs() < 0.05,
        "cold edge should stay at 1 m, got {base}"
    );
    assert!(
        (hot - 2.0).abs() < 0.05,
        "hot edge should reach 2 m, got {hot}"
    );
}

#[test]
fn active_strain_contracts_the_rest_edge() {
    let contracted = pinned_edge_length(|sb| assert!(sb.set_active_strain(0.5)), 120);
    // rest 1 m × (1 − 0.5) = 0.5 m.
    assert!(
        (contracted - 0.5).abs() < 0.05,
        "active strain 0.5 should contract to 0.5 m, got {contracted}"
    );
}
