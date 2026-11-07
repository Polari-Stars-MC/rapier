//! Cross-thread-count / multi-run determinism for the force-container path.
//!
//! Guards the bit-identical determinism contract of the consumed build
//! (default features + `parallel`): a body holding **two or more force kinds**
//! must produce bit-identical results across independent runs and across
//! thread-pool sizes. This class of bug is invisible to clone-compare tests
//! (`HashMap` iteration order differs per map instance), so both runs here build
//! fresh worlds.

use rapier3d_f64::prelude::*;

/// Runs `steps` steps of a small scene where every body carries two persistent
/// force kinds (Thrust + Custom), plus a contact pair, and returns each body's
/// translation bits so two runs can be compared exactly.
fn simulate(steps: usize, num_threads: Option<usize>) -> Vec<(u64, u64, u64)> {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut islands = IslandManager::new();
    let mut broad_phase = DefaultBroadPhase::new();
    let mut narrow_phase = NarrowPhase::new();
    let mut impulse_joints = ImpulseJointSet::new();
    let mut multibody_joints = MultibodyJointSet::new();
    let mut soft_bodies = SoftBodySet::new();
    let mut ccd = CCDSolver::new();
    let mut params = IntegrationParameters::default();

    let ground = bodies.insert(RigidBodyBuilder::fixed());
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(50.0, 0.5, 50.0),
        ground,
        &mut bodies,
    );

    let mut handles = Vec::new();
    for i in 0..4 {
        let h = bodies.insert(RigidBodyBuilder::dynamic().translation(Vector::new(
            i as f64 * 0.4 - 0.6,
            1.0 + i as f64,
            0.0,
        )));
        colliders.insert_with_parent(ColliderBuilder::ball(0.25), h, &mut bodies);
        handles.push(h);
    }

    let mut pipeline = PhysicsPipeline::new();
    if let Some(n) = num_threads {
        pipeline.configure_thread_pool(n).expect("thread pool");
    }

    let thrust = Vector::new(1.0, 2.5, -0.5);
    for (k, &h) in handles.iter().enumerate() {
        let rb = bodies.get_mut(h).unwrap();
        // Two distinct persistent kinds on one body: the effective-force sum must
        // not depend on container iteration order.
        rb.add_thrust(
            1,
            thrust,
            AngVector::ZERO,
            None,
            Persistence::Persistent,
            false,
        );
        rb.force_container_mut_with(ForceKind::Custom(7), Persistence::Persistent)
            .push(ForceEntry {
                id: 1,
                force: Vector::new(-0.5, 0.25, 0.75),
                torque: AngVector::new(0.1, 0.0, -0.05),
                point: Some(Vector::new(0.1, 0.0, 0.0)),
            });
        rb.add_torque(AngVector::new(0.3 * k as Real, 0.2, -0.1), false);
    }

    for _ in 0..steps {
        pipeline.step(
            Vector::new(0.0, -9.81, 0.0),
            &params,
            &mut islands,
            &mut broad_phase,
            &mut narrow_phase,
            &mut bodies,
            &mut colliders,
            &mut impulse_joints,
            &mut multibody_joints,
            &mut soft_bodies,
            &mut ccd,
            &(),
            &(),
        );
    }

    handles
        .iter()
        .map(|&h| {
            let t = bodies[h].translation();
            (t.x.to_bits(), t.y.to_bits(), t.z.to_bits())
        })
        .collect()
}

#[test]
fn force_containers_multi_kind_same_machine_determinism() {
    let a = simulate(120, None);
    let b = simulate(120, None);
    assert_eq!(
        a, b,
        "same scene, two fresh builds: results must be bit-identical"
    );
}

#[test]
fn force_containers_multi_kind_thread_count_determinism() {
    let single = simulate(120, Some(1));
    let four = simulate(120, Some(4));
    assert_eq!(
        single, four,
        "1-worker and 4-worker pools must produce bit-identical results"
    );
}
