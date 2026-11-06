//! The fork layer fused into the official soft body.
//!
//! These pin the solver-side wiring of `dynamics::soft_body::fork_extras`: wind is an
//! acceleration on every free particle, the fork damping is a per-step velocity scale applied
//! after the solve, thermal expansion and active strain rescale the edges' rest lengths, and
//! the orthotropic axes divide the per-edge compliance. Everything is inert by default, so
//! upstream's own soft-body tests stay the reference for the untouched path.

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

/// Two particles linked by one edge, the first one pinned.
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

fn mean_x(sb: &SoftBody) -> Real {
    let n = sb.num_particles() as Real;
    sb.particle_positions().map(|p| p.x).sum::<Real>() / n
}

fn mean_speed(sb: &SoftBody) -> Real {
    let n = sb.num_particles().max(1) as Real;
    sb.particle_velocities().map(|v| v.length()).sum::<Real>() / n
}

fn edge_length(sb: &SoftBody) -> Real {
    (sb.particle_position(1) - sb.particle_position(0)).length()
}

/// A world's worth of state, so the tests only spell out the body and the knobs.
struct Harness {
    bodies: RigidBodySet,
    colliders: ColliderSet,
    soft_bodies: SoftBodySet,
    pipeline: PhysicsPipeline,
    islands: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    ccd: CCDSolver,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    params: IntegrationParameters,
}

impl Harness {
    fn new() -> Self {
        Self {
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            soft_bodies: SoftBodySet::new(),
            pipeline: PhysicsPipeline::new(),
            islands: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            ccd: CCDSolver::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            params: IntegrationParameters::default(),
        }
    }

    fn step(&mut self, gravity: Vector, steps: usize) {
        for _ in 0..steps {
            self.pipeline.step(
                gravity,
                &self.params,
                &mut self.islands,
                &mut self.broad_phase,
                &mut self.narrow_phase,
                &mut self.bodies,
                &mut self.colliders,
                &mut self.impulse_joints,
                &mut self.multibody_joints,
                &mut self.soft_bodies,
                &mut self.ccd,
                &(),
                &(),
            );
        }
    }
}

#[test]
fn wind_accelerates_every_free_particle() {
    let mut h = Harness::new();
    let builder =
        SoftBodyBuilder::new(vec![Vector::new(0.0, 0.0, 0.0), Vector::new(0.0, 1.0, 0.0)])
            .edges(vec![[0, 1]])
            .no_surface_collider();
    let handle = h
        .soft_bodies
        .insert(builder, &mut h.bodies, &mut h.colliders);

    // 1 m/s² along +x, no drag.
    h.soft_bodies[handle].set_wind_field(Some(Wind {
        accel: Vector::new(1.0, 0.0, 0.0),
        drag: 0.0,
    }));
    h.step(Vector::ZERO, 60);

    let dt = h.params.dt;
    // Semi-implicit Euler: x = a·dt²·Σi = a·dt²·n(n+1)/2 for n = 60 steps of a = 1 m/s².
    let expected = dt * dt * (60.0 * 61.0 / 2.0);
    let x = mean_x(&h.soft_bodies[handle]);
    assert!(
        (x - expected).abs() < 0.02,
        "wind should integrate to {expected}, got {x}"
    );
}

#[test]
fn fork_damping_removes_velocity_every_step() {
    let mut speeds = Vec::new();

    for damping in [None, Some(0.2)] {
        let mut h = Harness::new();
        let builder =
            SoftBodyBuilder::new(vec![Vector::new(0.0, 0.0, 0.0), Vector::new(0.0, 1.0, 0.0)])
                .edges(vec![[0, 1]])
                .no_surface_collider();
        let handle = h
            .soft_bodies
            .insert(builder, &mut h.bodies, &mut h.colliders);

        let sb = &mut h.soft_bodies[handle];
        sb.set_particle_velocity(0, Vector::new(1.0, 0.0, 0.0));
        sb.set_particle_velocity(1, Vector::new(1.0, 0.0, 0.0));
        if let Some(d) = damping {
            assert!(sb.set_fork_damping(d), "0.2 is a valid damping");
        }

        h.step(Vector::ZERO, 30);
        speeds.push(mean_speed(&h.soft_bodies[handle]));
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
fn thermal_expansion_lengthens_the_rest_edge() {
    let cold = {
        let mut h = Harness::new();
        let (bodies, colliders, mut soft_bodies, handle) = pinned_two_particle_body();
        h.bodies = bodies;
        h.colliders = colliders;
        h.soft_bodies = std::mem::take(&mut soft_bodies);
        h.step(Vector::ZERO, 120);
        edge_length(&h.soft_bodies[handle])
    };
    let hot = {
        let mut h = Harness::new();
        let (bodies, colliders, mut soft_bodies, handle) = pinned_two_particle_body();
        h.bodies = bodies;
        h.colliders = colliders;
        h.soft_bodies = std::mem::take(&mut soft_bodies);
        assert!(h.soft_bodies[handle].set_thermal_field(Some(ThermalParams {
            temp: 100.0,
            ambient: 0.0,
            expansion: 0.01,
            stiffness_temp_coeff: 0.0,
        })));
        h.step(Vector::ZERO, 120);
        edge_length(&h.soft_bodies[handle])
    };

    // rest 1 m, and 1 m × (1 + 0.01·100) = 2 m when hot.
    assert!(
        (cold - 1.0).abs() < 0.05,
        "cold edge should stay at 1 m, got {cold}"
    );
    assert!(
        (hot - 2.0).abs() < 0.05,
        "hot edge should reach 2 m, got {hot}"
    );
}

#[test]
fn active_strain_contracts_the_rest_edge() {
    let mut h = Harness::new();
    let (bodies, colliders, soft_bodies, handle) = pinned_two_particle_body();
    h.bodies = bodies;
    h.colliders = colliders;
    h.soft_bodies = soft_bodies;
    assert!(h.soft_bodies[handle].set_active_strain(0.5));
    h.step(Vector::ZERO, 120);

    let len = edge_length(&h.soft_bodies[handle]);
    // rest 1 m × (1 − 0.5) = 0.5 m.
    assert!(
        (len - 0.5).abs() < 0.05,
        "active strain 0.5 should contract to 0.5 m, got {len}"
    );
}
