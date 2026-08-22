use crate::alloc_prelude::*;
use crate::dynamics::{IntegrationParameters, IslandManager, RigidBodySet};
use crate::geometry::{
    BroadPhaseBvh, Collider, ColliderHandle, ColliderSet, CollisionEvent, NarrowPhase,
};
use crate::math::Real;
use crate::parry::bounding_volume::Aabb;
use crate::pipeline::{EventHandler, PhysicsHooks, QueryFilter};
use crate::prelude::{ActiveEvents, CollisionEventFlags};
use parry::query::sweep_toi::Sweep;

use super::sweeps::{
    BodyContinuousResult, CcdTargets, PseudoHitMode, collect_fixed_targets, is_bullet,
    map_bodies_parallel, sweep_fast_body,
};

/// Continuous Collision Detection solver preventing fast objects from tunneling:
/// after the solver, bodies that moved more than half their thinnest extent sweep their colliders
/// and `next_position` is clamped to the earliest impact — velocities untouched, no re-solve; the
/// residual approach resolves next step via speculative contacts.
///
/// Fast dynamic bodies automatically sweep against **fixed** colliders; `ccd_enabled` upgrades to
/// a *bullet* that also sweeps kinematic/dynamic bodies (including other bullets — issue #984). Mesh-like colliders
/// are never swept as the *moving* shape (targets are fine), compounds sweep per
/// convex child, and [`IntegrationParameters::max_ccd_substeps`] `= 0` disables CCD entirely.
#[derive(Clone, Default)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct CCDSolver {
    /// Cached fixed-target list for the non-bullet sweep pass: the AABB loosening it was built
    /// with; `None` past [`FIXED_TARGETS_LIST_MAX`] (sweep queries the full BVH). Invalidated by
    /// the scene-change flag — re-scanning every collider each step dominated CCD on large scenes.
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    fixed_targets_cache: Option<FixedTargetsCache>,
}

/// The AABB loosening the cached fixed-target list was built with, paired with the list
/// itself — `None` past [`FIXED_TARGETS_LIST_MAX`], where the sweep queries the full BVH.
type FixedTargetsCache = (Real, Option<Vec<(ColliderHandle, Aabb)>>);

impl CCDSolver {
    /// Initializes a new CCD solver
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates the set of bodies that needs CCD to be resolved.
    ///
    /// Returns `true` if any rigid-body must have CCD resolved.
    pub fn update_ccd_active_flags(
        &self,
        islands: &IslandManager,
        bodies: &mut RigidBodySet,
        dt: Real,
        include_forces: bool,
    ) -> bool {
        let mut ccd_active = false;

        for handle in islands.active_bodies() {
            let rb = bodies.index_mut_internal(handle);

            // Default tier: every fast dynamic body is a CCD origin. `ccd_enabled`
            // no longer gates *activation*, only the sweep *scope* (fixed-only vs all bodies),
            // applied later during pair selection.
            if rb.is_dynamic() {
                let moving_fast = if include_forces {
                    // Pre-solve (substep splitter): `next_position` isn't solved yet, use
                    // the velocity-based estimate including forces.
                    rb.ccd.is_moving_fast(
                        dt,
                        &rb.ccd_vels,
                        Some(&rb.forces),
                        rb.mprops.max_extent(),
                    )
                } else {
                    // Post-solve: the fast-body criterion on the actual solved motion.
                    rb.ccd.is_moving_fast_with_next_position(
                        dt,
                        &rb.ccd_vels,
                        &rb.pos,
                        rb.mprops.local_mprops.local_com,
                        rb.mprops.max_extent(),
                    )
                };
                rb.ccd.ccd_active = moving_fast;
                ccd_active = ccd_active || moving_fast;
            }
        }

        ccd_active
    }

    /// Find the first time a CCD-active body has a non-sensor collider hitting another
    /// non-sensor collider, for the multi-substep splitter.
    ///
    /// Returns the impact time in `[0, dt)` if any.
    #[profiling::function]
    #[allow(clippy::too_many_arguments)]
    pub fn find_first_impact(
        &mut self,
        dt: Real, // NOTE: this doesn’t necessarily match the `params.dt`.
        params: &IntegrationParameters,
        islands: &IslandManager,
        bodies: &RigidBodySet,
        colliders: &ColliderSet,
        broad_phase: &mut BroadPhaseBvh,
        narrow_phase: &NarrowPhase,
        hooks: &dyn PhysicsHooks,
    ) -> Option<Real> {
        // NOTE: broad-phase AABBs are NOT enlarged to the swept volumes: only the fast body's
        // query box is swept (per collider, below); targets keep their regular fat AABBs.
        // Swept AABBs written into the tree would leak into the next step (pair explosion).
        let query_pipeline = broad_phase.as_query_pipeline(
            narrow_phase.query_dispatcher(),
            bodies,
            colliders,
            QueryFilter::default(),
        );
        let (bvh, dispatcher) = (query_pipeline.bvh, query_pipeline.dispatcher);

        let linear_slop = params.allowed_linear_error();
        let fast_bodies: Vec<_> = islands
            .active_bodies()
            .filter(|h| bodies[*h].ccd.ccd_active)
            .collect();

        let fractions = map_bodies_parallel(&fast_bodies, hooks, |handle, hooks| {
            let rb1 = &bodies[handle];
            // `next_position` isn't solved yet: sweep to the forces/velocities integration.
            let predicted_body_pos =
                rb1.pos
                    .integrate_forces_and_velocities(dt, &rb1.forces, &rb1.vels, &rb1.mprops);
            sweep_fast_body(
                handle,
                bodies,
                colliders,
                predicted_body_pos,
                CcdTargets::FullBvh(bvh),
                dispatcher,
                hooks,
                dt,
                linear_slop,
                PseudoHitMode::Ignore,
            )
            .fraction
        });

        let min_fraction = fractions.into_iter().fold(1.0, Real::min);
        (min_fraction < 1.0).then_some(min_fraction * dt)
    }

    /// Runs the continuous-collision pass on all fast bodies and clamps their `next_position`
    /// to their earliest time of impact: non-bullets sweep fixed colliders first, then bullets
    /// sweep every (possibly already clamped) body; velocities are never modified. Sensor
    /// crossings the narrow phase would miss entirely emit paired `Started`/`Stopped`
    /// intersection events.
    #[profiling::function]
    #[allow(clippy::too_many_arguments)]
    pub fn solve_continuous(
        &mut self,
        params: &IntegrationParameters,
        islands: &IslandManager,
        bodies: &mut RigidBodySet,
        colliders: &ColliderSet,
        broad_phase: &mut BroadPhaseBvh,
        narrow_phase: &NarrowPhase,
        hooks: &dyn PhysicsHooks,
        events: &dyn EventHandler,
        // `true` when colliders/bodies were added, removed or modified by the
        // user since the last step: the only ways a fixed target can appear,
        // vanish or move, hence the fixed-target cache invalidation signal.
        scene_changed: bool,
    ) {
        let dt = params.dt;
        let linear_slop = params.allowed_linear_error();

        // NOTE: broad-phase AABBs are NOT enlarged to the swept volumes: only the fast body's
        // query box is swept; stationary targets keep their fat AABBs. Swept AABBs in
        // the tree leak into the next step's broad phase (pair explosion, ~2x narrow-phase cost).
        let (non_bullets, bullets): (Vec<_>, Vec<_>) = islands
            .active_bodies()
            .filter(|h| bodies[*h].ccd.ccd_active)
            .partition(|h| !is_bullet(&bodies[*h]));


        let mut all_results = Vec::new();

        // Pass 1: fast non-bullet bodies vs fixed targets (all targets are stationary, so
        // the bodies are independent and can run in parallel).
        {
            let query_pipeline = broad_phase.as_query_pipeline(
                narrow_phase.query_dispatcher(),
                bodies,
                colliders,
                QueryFilter::default(),
            );
            let (bvh, dispatcher) = (query_pipeline.bvh, query_pipeline.dispatcher);
            // Non-bullet fast bodies only hit fixed targets: sweep against the (small) cached
            // fixed-collider list instead of the full BVH. Rebuilt — a full collider scan —
            // only on scene changes, since fixed targets can't move otherwise.
            let prediction = params.prediction_distance();
            let cache_valid = !scene_changed
                && self
                    .fixed_targets_cache
                    .as_ref()
                    .is_some_and(|(p, _)| *p == prediction);
            if !cache_valid {
                self.fixed_targets_cache = Some((
                    prediction,
                    collect_fixed_targets(bodies, colliders, prediction),
                ));
            }
            let targets = match &self.fixed_targets_cache.as_ref().unwrap().1 {
                Some(fixed) => CcdTargets::FixedList(fixed),
                None => CcdTargets::FullBvh(bvh),
            };
            let results = map_bodies_parallel(&non_bullets, hooks, |handle, hooks| {
                sweep_fast_body(
                    handle,
                    bodies,
                    colliders,
                    bodies[handle].pos.next_position,
                    targets,
                    dispatcher,
                    hooks,
                    dt,
                    linear_slop,
                    PseudoHitMode::Record,
                )
            });
            all_results.extend(results);
        }
        Self::apply_clamps(bodies, &all_results);

        // Pass 2: bullets vs every (possibly already clamped) body, including other bullets.
        // clamped) `next_position` from pass 1 (deferred bullet stage).
        if !bullets.is_empty() {
            let bullet_results = {
                let query_pipeline = broad_phase.as_query_pipeline(
                    narrow_phase.query_dispatcher(),
                    bodies,
                    colliders,
                    QueryFilter::default(),
                );
                let (bvh, dispatcher) = (query_pipeline.bvh, query_pipeline.dispatcher);
                map_bodies_parallel(&bullets, hooks, |handle, hooks| {
                    sweep_fast_body(
                        handle,
                        bodies,
                        colliders,
                        bodies[handle].pos.next_position,
                        CcdTargets::FullBvh(bvh),
                        dispatcher,
                        hooks,
                        dt,
                        linear_slop,
                        PseudoHitMode::Record,
                    )
                })
            };
            Self::apply_clamps(bodies, &bullet_results);
            all_results.extend(bullet_results);
        }

        // Emit intersection events for sensor crossings that happened strictly before each
        // body's final solid impact and that the narrow phase would never observe (no
        // overlap at either the start or the clamped end pose).
        for result in &all_results {
            for hit in &result.pseudo_hits {
                if hit.fraction >= result.fraction {
                    // The body stops before reaching this sensor.
                    continue;
                }

                let co1 = &colliders[hit.ch1];
                let co2 = &colliders[hit.ch2];

                if !co1.is_sensor() && !co2.is_sensor() {
                    // TODO: this happens if we found a TOI between two non-sensor
                    //       colliders with mismatching solver_flags. It is not clear
                    //       what we should do in this case: we could report a
                    //       contact started/contact stopped event for example. But in
                    //       that case, what contact pair should be pass to these events?
                    // For now we just ignore this special case. Let's wait for an actual
                    // use-case to come up before we determine what we want to do here.
                    continue;
                }

                let next_pose = |co: &Collider| match co.parent.as_ref() {
                    Some(parent) => bodies[parent.handle].pos.next_position * parent.pos_wrt_parent,
                    None => co.pos.0,
                };

                let prev_pos12 = co1.pos.inv_mul(&co2.pos);
                let next_pos12 = next_pose(co1).inv_mul(&next_pose(co2));

                let dispatcher = narrow_phase.query_dispatcher();
                let intersect_before = dispatcher
                    .intersection_test(&prev_pos12, co1.shape.as_ref(), co2.shape.as_ref())
                    .unwrap_or(false);
                let intersect_after = dispatcher
                    .intersection_test(&next_pos12, co1.shape.as_ref(), co2.shape.as_ref())
                    .unwrap_or(false);

                if !intersect_before
                    && !intersect_after
                    && (co1.flags.active_events | co2.flags.active_events)
                        .contains(ActiveEvents::COLLISION_EVENTS)
                {
                    // Emit one intersection-started and one intersection-stopped event.
                    events.handle_collision_event(
                        bodies,
                        colliders,
                        CollisionEvent::Started(hit.ch1, hit.ch2, CollisionEventFlags::SENSOR),
                        None,
                    );
                    events.handle_collision_event(
                        bodies,
                        colliders,
                        CollisionEvent::Stopped(hit.ch1, hit.ch2, CollisionEventFlags::SENSOR),
                        None,
                    );
                }
            }
        }
    }

    /// Clamps each impacted body's `next_position` to the interpolated pose at its impact
    /// fraction. Pose only — velocities are preserved.
    fn apply_clamps(bodies: &mut RigidBodySet, results: &[BodyContinuousResult]) {
        for result in results {
            if result.fraction < 1.0 {
                let rb = bodies.index_mut_internal(result.handle);
                let sweep = Sweep::from_poses(
                    &rb.pos.position,
                    &rb.pos.next_position,
                    rb.mprops.local_mprops.local_com,
                );
                rb.pos.next_position = sweep.transform_at(result.fraction);
                // Record the point of impact (issue #548) from the earliest solid hit.
                if let Some(toi) = &result.toi {
                    rb.ccd.toi_point = toi.point;
                    rb.ccd.toi_normal = toi.normal;
                }
            }
        }
    }
}


#[cfg(all(feature = "dim3"))]
#[cfg(test)]
mod ccd_tests {
    use crate::dynamics::{ImpulseJointSet, IslandManager, MultibodyJointSet, RigidBodySet};
    use crate::geometry::{ColliderSet, NarrowPhase};
    use crate::math::Vector;
    use std::vec::Vec;
    use crate::prelude::{
        CCDSolver, ColliderBuilder, DefaultBroadPhase, IntegrationParameters, PhysicsPipeline,
        RigidBodyBuilder,
    };

    /// Regression test for #984 (CCD tunneling between two fast bodies).
    ///
    /// Two dynamic bullets approach head-on at 100 m/s. After one step a raycast along the
    /// center line must hit the OTHER bullet before reaching its starting body (no tunnel).
    #[test]
    fn ccd_between_two_fast_bullets_does_not_tunnel() {
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let mut island_manager = IslandManager::new();
        let mut broad_phase = DefaultBroadPhase::new();
        let mut narrow_phase = NarrowPhase::new();
        let mut impulse_joint_set = ImpulseJointSet::new();
        let mut multibody_joint_set = MultibodyJointSet::new();
        let mut ccd_solver = CCDSolver::new();
        let mut physics_pipeline = PhysicsPipeline::new();
        let gravity = Vector::new(0.0, 0.0, 0.0);
        let integration_parameters = IntegrationParameters::default();

        // Two bullets about to collide: A at -0.7 moving +x, B at +0.7 moving -x, each 100 m/s.
        // Spacing is within the swept-AABB reach so the broad phase emits the candidate pair.
        let a = RigidBodyBuilder::dynamic()
            .translation(Vector::new(-0.7, 0.0, 0.0))
            .ccd_enabled(true)
            .build();
        let a_h = bodies.insert(a);
        let b = RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.7, 0.0, 0.0))
            .ccd_enabled(true)
            .build();
        let b_h = bodies.insert(b);
        colliders.insert_with_parent(ColliderBuilder::ball(0.5), a_h, &mut bodies);
        colliders.insert_with_parent(ColliderBuilder::ball(0.5), b_h, &mut bodies);

        // One real step populates broad/narrow phase + islands; then we overwrite the solved
        // poses to the *crossing* configuration and force CCD active, calling solve_continuous
        // directly to exercise the bullet-vs-bullet sweep (issue #984).
        physics_pipeline.step(
            gravity,
            &integration_parameters,
            &mut island_manager,
            &mut broad_phase,
            &mut narrow_phase,
            &mut bodies,
            &mut colliders,
            &mut impulse_joint_set,
            &mut multibody_joint_set,
            &mut ccd_solver,
            &(),
            &(),
        );

        let a_mut = bodies.get_mut(a_h).unwrap();
        a_mut.pos.position.translation.x = -0.7;
        a_mut.pos.next_position.translation.x = 0.5; // naive integration would cross past B
        a_mut.ccd.ccd_active = true;
        let b_mut = bodies.get_mut(b_h).unwrap();
        b_mut.pos.position.translation.x = 0.7;
        b_mut.pos.next_position.translation.x = -0.5; // naive integration would cross past A
        b_mut.ccd.ccd_active = true;

        broad_phase.update(
            &integration_parameters,
            &colliders,
            &bodies,
            &[],
            &[],
            &mut Vec::new(),
        );

        ccd_solver.solve_continuous(
            &integration_parameters,
            &island_manager,
            &mut bodies,
            &colliders,
            &mut broad_phase,
            &narrow_phase,
            &(),
            &(),
            true,
        );

        let a_pos = bodies.get(a_h).unwrap().pos.next_position.translation.x;
        let b_pos = bodies.get(b_h).unwrap().pos.next_position.translation.x;

        assert!(
            a_pos < 0.0 && b_pos > 0.0 && a_pos < b_pos,
            "bullets tunneled through each other: A.x={}, B.x={}",
            a_pos,
            b_pos
        );
    }

    /// Regression test for #548 (CCD point of impact).
    ///
    /// After the bullet-vs-bullet solve, each body must record a world-space point of impact
    /// and a separating normal (the relative axis between the two balls, i.e. ∓X here).
    #[test]
    fn ccd_records_point_of_impact() {
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let mut island_manager = IslandManager::new();
        let mut broad_phase = DefaultBroadPhase::new();
        let mut narrow_phase = NarrowPhase::new();
        let mut impulse_joint_set = ImpulseJointSet::new();
        let mut multibody_joint_set = MultibodyJointSet::new();
        let mut ccd_solver = CCDSolver::new();
        let mut physics_pipeline = PhysicsPipeline::new();
        let gravity = Vector::new(0.0, 0.0, 0.0);
        let integration_parameters = IntegrationParameters::default();

        let a = RigidBodyBuilder::dynamic()
            .translation(Vector::new(-0.7, 0.0, 0.0))
            .ccd_enabled(true)
            .build();
        let a_h = bodies.insert(a);
        let b = RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.7, 0.0, 0.0))
            .ccd_enabled(true)
            .build();
        let b_h = bodies.insert(b);
        colliders.insert_with_parent(ColliderBuilder::ball(0.5), a_h, &mut bodies);
        colliders.insert_with_parent(ColliderBuilder::ball(0.5), b_h, &mut bodies);

        physics_pipeline.step(
            gravity,
            &integration_parameters,
            &mut island_manager,
            &mut broad_phase,
            &mut narrow_phase,
            &mut bodies,
            &mut colliders,
            &mut impulse_joint_set,
            &mut multibody_joint_set,
            &mut ccd_solver,
            &(),
            &(),
        );

        let a_mut = bodies.get_mut(a_h).unwrap();
        a_mut.pos.position.translation.x = -0.7;
        a_mut.pos.next_position.translation.x = 0.5;
        a_mut.ccd.ccd_active = true;
        let b_mut = bodies.get_mut(b_h).unwrap();
        b_mut.pos.position.translation.x = 0.7;
        b_mut.pos.next_position.translation.x = -0.5;
        b_mut.ccd.ccd_active = true;

        broad_phase.update(
            &integration_parameters,
            &colliders,
            &bodies,
            &[],
            &[],
            &mut Vec::new(),
        );

        ccd_solver.solve_continuous(
            &integration_parameters,
            &island_manager,
            &mut bodies,
            &colliders,
            &mut broad_phase,
            &narrow_phase,
            &(),
            &(),
            true,
        );

        let a_toi = bodies.get(a_h).unwrap().ccd_point_of_impact();
        let b_toi = bodies.get(b_h).unwrap().ccd_point_of_impact();
        assert!(a_toi.is_some(), "A must record a CCD point of impact");
        assert!(b_toi.is_some(), "B must record a CCD point of impact");

        let (a_point, a_normal) = a_toi.unwrap();
        let (b_point, b_normal) = b_toi.unwrap();
        // The hit point lies on the A-B center line (x-axis), y=z=0.
        assert!(
            a_point.y.abs() < 1.0e-6 && a_point.z.abs() < 1.0e-6,
            "impact point must lie on the A-B center line, got {:?}",
            a_point
        );
        // The separating normals are along ±X (the A↔B axis) and oppose each other: one body
        // records +X, the other -X, since the normal is the relative separating axis.
        assert!(
            a_normal.x.abs() > 0.99 && b_normal.x.abs() > 0.99,
            "both normals must align with X, got A={:?} B={:?}",
            a_normal,
            b_normal
        );
        assert!(
            a_normal.x * b_normal.x < 0.0,
            "A and B separating normals must oppose, got A={:?} B={:?}",
            a_normal,
            b_normal
        );
        // The two records describe the same contact point.
        assert!(
            (a_point.x - b_point.x).abs() < 1.0e-6,
            "A and B must agree on the impact x, got {} vs {}",
            a_point.x,
            b_point.x
        );
    }
}

