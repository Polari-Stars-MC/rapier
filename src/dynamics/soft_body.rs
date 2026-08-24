//! Soft-body (deformable body) support — Phase 0 foundation.
//!
//! This module is the **data-structure + free-particle integrator** skeleton for
//! soft-body simulation. It is intentionally independent of the SoA SIMD solver
//! boundary (`helpers.rs` / `worker.rs` / `generic_contact_constraint.rs` read
//! the `velocities` / `accelerations` nalgebra buffers) so it can be compiled and
//! unit-tested in isolation.
//!
//! ## Design notes (see `.hermes/plans/2026-08-24_soft-body-roadmap.md`)
//!
//! * A soft body is a cloud of point masses (`SoftParticle`) connected by
//!   Hookean springs (`Spring`). Springs carry `stiffness` (`k`) and `damping`
//!   (`c`); the spring force is `F = -k·(|x_b - x_a| - rest)·dir - c·(v_rel · dir)·dir`.
//! * Integration is **semi-implicit (symplectic) Euler**: velocities are updated
//!   first (`v += dt · M⁻¹ · f`), then positions (`x += dt · v`). This matches the
//!   operation order used by the rigid-body integrator, and keeps the floating-point
//!   sequence bit-identical across runs under `enhanced-determinism`.
//! * The internal spring/damping forces are the natural payload for the existing
//!   `force_containers` `ForceKind::Custom(Persistent)` model (Phase 2 will route
//!   them there); Phase 0 keeps them local so the numerics can be tested directly.
//! * All math uses `crate::math::Vector` (glam-backed) and `Real` so the module
//!   stays aligned with the rest of the fork's vector conventions.
//!
//! Phase 0a (this file) = data structures + integrator + tests. Phase 0b (later)
//! wires `SoftBodySet` into `World` / `PersistentIslands`. Phase 1+ add joint-based
//! and mass-spring/FEM coupling.

use crate::dynamics::{
    RigidBodyHandle, RigidBodySet,
    force_containers::{ForceEntry, ForceKind, KindContainer, Persistence},
};
use crate::math::{AngVector, Real, Vector};
use std::vec::Vec;

/// `ForceKind::Custom` discriminator for soft-body internal spring/damping forces.
/// Routed through `force_containers` so they share the same `Persistent` lifecycle
/// and `compute_body_effective_forces` summation as gravity/thrust (Phase 2).
pub const SOFT_SPRING_CUSTOM_ID: u32 = 0x5_042; // "SB" encoded; arbitrary custom tag

/// Opaque id of a [`SoftBody`] inside a [`SoftBodySet`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SoftBodyId(pub u32);

/// A single point mass in a soft body.
#[derive(Clone, Debug)]
pub struct SoftParticle {
    /// Current world-space position.
    pub pos: Vector,
    /// Current world-space linear velocity.
    pub vel: Vector,
    /// Accumulated force for the current substep (cleared by `compute_forces`).
    pub force: Vector,
    /// Inverse mass (`0` = pinned / immovable). `mass = 1 / inv_mass`.
    pub inv_mass: Real,
    /// Optional rigid body this particle is bound to. When set, the particle's
    /// internal spring/damping force is routed into that body's `force_containers`
    /// (Phase 2), so the soft body drives the rigid body through the standard
    /// effective-force path rather than integrating the particle directly.
    pub bound_body: Option<RigidBodyHandle>,
}

impl SoftParticle {
    /// Creates a free (movable) particle with unit mass.
    pub fn new(pos: Vector) -> Self {
        Self {
            pos,
            vel: Vector::ZERO,
            force: Vector::ZERO,
            inv_mass: 1.0,
            bound_body: None,
        }
    }

    /// Creates a pinned (immovable) particle — its `inv_mass` is `0`.
    pub fn pinned(pos: Vector) -> Self {
        Self {
            pos,
            vel: Vector::ZERO,
            force: Vector::ZERO,
            inv_mass: 0.0,
            bound_body: None,
        }
    }

    /// Mass of the particle; `0` for pinned particles.
    #[inline]
    pub fn mass(&self) -> Real {
        if self.inv_mass == 0.0 {
            0.0
        } else {
            1.0 / self.inv_mass
        }
    }
}

/// A Hookean spring connecting two particles, with linear damping along its axis.
#[derive(Clone, Copy, Debug)]
pub struct Spring {
    /// Index of the first endpoint particle within the parent [`SoftBody`].
    pub a: usize,
    /// Index of the second endpoint particle within the parent [`SoftBody`].
    pub b: usize,
    /// Rest length (natural length) of the spring.
    pub rest_length: Real,
    /// Spring constant `k` (stiffness).
    pub stiffness: Real,
    /// Damping coefficient `c` (axial velocity damping).
    pub damping: Real,
}

/// A soft body: a collection of point masses connected by springs.
///
/// In Phase 0 it is a standalone structure. Later phases attach it to a
/// [`crate::dynamics::SoftBodySet`] and (optionally) to the solver islands.
#[derive(Clone, Debug)]
pub struct SoftBody {
    /// Point masses. Spring endpoints index into this `Vec`.
    pub particles: Vec<SoftParticle>,
    /// Springs (edges) between particles.
    pub springs: Vec<Spring>,
    /// Constant acceleration applied to every free particle (typically gravity).
    pub gravity: Vector,
    /// Coarse sleeping flag (island-style). When `true`, [`SoftBody::step`] is a
    /// no-op: the whole body is treated as an inactive unit, mirroring how
    /// `PersistentIslands` keeps a sleeping rigid-body island from being
    /// re-simulated. Per-particle island membership is Phase 3; this flag gives
    /// the same "skip inactive work" behavior at body granularity for now.
    pub sleeping: bool,
}

impl SoftBody {
    /// Creates an empty soft body in a gravity field `gravity`.
    pub fn new(gravity: Vector) -> Self {
        Self {
            particles: Vec::new(),
            springs: Vec::new(),
            gravity,
            sleeping: false,
        }
    }

    /// Adds a free particle and returns its index.
    pub fn add_particle(&mut self, pos: Vector) -> usize {
        let idx = self.particles.len();
        self.particles.push(SoftParticle::new(pos));
        idx
    }

    /// Adds a pinned (immovable) particle and returns its index.
    pub fn add_pinned(&mut self, pos: Vector) -> usize {
        let idx = self.particles.len();
        self.particles.push(SoftParticle::pinned(pos));
        idx
    }

    /// Adds a spring between particles `a` and `b` with the given stiffness/damping.
    /// The rest length is taken from the current distance between the endpoints.
    /// Returns `None` (and does nothing) if either index is out of bounds or the
    /// endpoints coincide (zero rest length is rejected to avoid a degenerate axis).
    pub fn add_spring(
        &mut self,
        a: usize,
        b: usize,
        stiffness: Real,
        damping: Real,
    ) -> Option<usize> {
        let (pa, pb) = (self.particles.get(a)?, self.particles.get(b)?);
        let rest = (pb.pos - pa.pos).length();
        if rest == 0.0 {
            return None;
        }
        let idx = self.springs.len();
        self.springs.push(Spring {
            a,
            b,
            rest_length: rest,
            stiffness,
            damping,
        });
        Some(idx)
    }

    /// Computes the per-particle internal force from all springs (Hookean +
    /// axial damping), *without* gravity. Returns one `Vector` per particle.
    /// Pinned particles (`inv_mass == 0`) receive no force (they act as anchors).
    ///
    /// This is the force that Phase 2 routes into `force_containers` for bound
    /// particles; `compute_forces` reuses it and adds gravity for free particles.
    pub fn spring_damping_forces(&self) -> Vec<Vector> {
        let mut out = Vec::with_capacity(self.particles.len());
        for _ in 0..self.particles.len() {
            out.push(Vector::ZERO);
        }
        for s in &self.springs {
            let (pa_pos, pb_pos, pa_vel, pb_vel, pa_im, pb_im) =
                match (self.particles.get(s.a), self.particles.get(s.b)) {
                    (Some(a), Some(b)) => (a.pos, b.pos, a.vel, b.vel, a.inv_mass, b.inv_mass),
                    _ => continue,
                };
            let delta = pb_pos - pa_pos;
            let len = delta.length();
            if len == 0.0 {
                continue;
            }
            let dir = delta / len;
            let f_spring = s.stiffness * (len - s.rest_length);
            let rel_vel = pb_vel - pa_vel;
            let f_damp = s.damping * rel_vel.dot(dir);
            let f_axial = f_spring + f_damp;
            let f = dir * f_axial;
            if pa_im != 0.0 {
                out[s.a] += f;
            }
            if pb_im != 0.0 {
                out[s.b] -= f;
            }
        }
        out
    }

    /// Accumulates the total force on every particle: gravity plus all spring
    /// (Hookean + axial damping) contributions. Clears each particle's `force`
    /// first, so this can be called once per substep before `integrate`.
    ///
    /// Bound particles (those with `bound_body` set) do **not** accumulate a local
    /// force here — their spring force is instead routed to the rigid body via
    /// [`SoftBodySet::write_spring_forces`](crate::dynamics::SoftBodySet::write_spring_forces),
    /// so the rigid body's own integrator applies it through `force_containers`.
    pub fn compute_forces(&mut self) {
        let spring = self.spring_damping_forces();
        for (i, p) in self.particles.iter_mut().enumerate() {
            if p.bound_body.is_some() {
                // Driven externally via force_containers; no local integration.
                p.force = Vector::ZERO;
                continue;
            }
            p.force = if p.inv_mass == 0.0 {
                Vector::ZERO
            } else {
                p.mass() * self.gravity
            };
            p.force += spring[i];
        }
    }

    /// Advances velocities then positions by `dt` (semi-implicit Euler).
    /// Pinned particles (`inv_mass == 0`) are not moved.
    pub fn integrate(&mut self, dt: Real) {
        for p in &mut self.particles {
            if p.inv_mass == 0.0 {
                continue;
            }
            // v += dt · M⁻¹ · f   (M⁻¹ = inv_mass for a point mass)
            p.vel += dir_scaled(p.force, dt * p.inv_mass);
            // x += dt · v
            p.pos += dir_scaled(p.vel, dt);
        }
    }

    /// One substep: clear/accumulate forces, then integrate.
    pub fn step(&mut self, dt: Real) {
        if self.sleeping {
            return;
        }
        self.compute_forces();
        self.integrate(dt);
    }

    /// Total kinetic energy of the free particles (`½ · m · |v|²`).
    pub fn kinetic_energy(&self) -> Real {
        self.particles
            .iter()
            .filter(|p| p.inv_mass != 0.0)
            .map(|p| 0.5 * p.mass() * p.vel.dot(p.vel))
            .fold(0.0, |acc, e| acc + e)
    }
}

/// `v * s` for a `Vector` and scalar `s` (glam supports `Vec3 * f64`).
#[inline]
fn dir_scaled(v: Vector, s: Real) -> Vector {
    v * s
}

/// A container owning all soft bodies in a simulation. Phase 0 keeps this as a
/// plain `Vec` store; later phases may back it with the arena used by the
/// rigid-body / joint sets.
#[derive(Clone, Debug, Default)]
pub struct SoftBodySet {
    bodies: Vec<SoftBody>,
}

impl SoftBodySet {
    /// Creates an empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts a soft body and returns its id.
    pub fn insert(&mut self, body: SoftBody) -> SoftBodyId {
        let id = SoftBodyId(self.bodies.len() as u32);
        self.bodies.push(body);
        id
    }

    /// Number of soft bodies.
    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }

    /// Immutable access by id.
    #[allow(dead_code)] // consumed by later integration phases (World/FFI).
    pub fn get(&self, id: SoftBodyId) -> Option<&SoftBody> {
        self.bodies.get(id.0 as usize)
    }

    /// Mutable access by id.
    pub fn get_mut(&mut self, id: SoftBodyId) -> Option<&mut SoftBody> {
        self.bodies.get_mut(id.0 as usize)
    }

    /// Advances every soft body by `dt` (sleeping bodies are skipped).
    pub fn step(&mut self, dt: Real) {
        for body in &mut self.bodies {
            body.step(dt);
        }
    }

    /// Marks a soft body as sleeping (no further integration until woken).
    pub fn sleep(&mut self, id: SoftBodyId) -> bool {
        if let Some(b) = self.bodies.get_mut(id.0 as usize) {
            b.sleeping = true;
            true
        } else {
            false
        }
    }

    /// Wakes a sleeping soft body.
    pub fn wake(&mut self, id: SoftBodyId) -> bool {
        if let Some(b) = self.bodies.get_mut(id.0 as usize) {
            b.sleeping = false;
            true
        } else {
            false
        }
    }

    /// Whether the soft body is currently sleeping.
    pub fn is_sleeping(&self, id: SoftBodyId) -> bool {
        self.bodies
            .get(id.0 as usize)
            .map(|b| b.sleeping)
            .unwrap_or(false)
    }

    /// Phase 2: routes each soft body's internal spring/damping forces into the
    /// `force_containers` of the rigid bodies their (bound) particles drive.
    ///
    /// For every particle with `bound_body = Some(h)`, its spring/damping force is
    /// written as a `ForceKind::Custom(SOFT_SPRING_CUSTOM_ID)` **Persistent**
    /// `ForceEntry` (application point = particle position, so off-center forces
    /// generate the correct `r × F` torque). The rigid body then receives the soft
    /// force through the standard `compute_body_effective_forces` path — no new
    /// solver code, identical lifecycle handling to gravity/thrust.
    ///
    /// The soft container for each body is cleared and rebuilt each call so the
    /// forces stay in sync with the current particle positions (a `Persistent`
    /// container survives frame-end draining, but we own it and overwrite it).
    /// Sleeping soft bodies are skipped entirely.
    pub fn write_spring_forces(&self, bodies: &mut RigidBodySet) {
        let kind = ForceKind::Custom(SOFT_SPRING_CUSTOM_ID);
        for body in &self.bodies {
            if body.sleeping {
                continue;
            }
            let spring = body.spring_damping_forces();
            for (i, p) in body.particles.iter().enumerate() {
                let Some(h) = p.bound_body else { continue };
                let rb = match bodies.get_mut(h) {
                    Some(rb) => rb,
                    None => continue,
                };
                // Clear + rebuild this body's soft-spring container.
                rb.force_containers.remove(&kind);
                let entry = ForceEntry {
                    id: i as u64 + 1,
                    force: spring[i],
                    torque: AngVector::ZERO,
                    point: Some(p.pos),
                };
                rb.force_containers
                    .entry(kind)
                    .or_insert_with(|| KindContainer::new(kind, Persistence::Persistent))
                    .push(entry);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — bit-identical numerics.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_fall_is_analytic() {
        // Semi-implicit (symplectic) Euler, single step, is bit-identical to the
        // closed form `x1 = x0 + dt·v0 + dt²·a`. We assert with a single step so the
        // floating-point operation sequence matches exactly (multi-step iteration
        // would round differently from a one-shot closed form and break to_bits).
        let g = Vector::new(0.0, -9.81, 0.0);
        let mut body = SoftBody::new(g);
        let x0 = Vector::new(0.0, 10.0, 0.0);
        body.add_particle(x0);
        body.particles[0].vel = Vector::new(1.0, 0.0, 0.0);

        let dt = 0.01;
        body.step(dt); // exactly one step

        // Closed form for n = 1:  v1 = v0 + dt·a ;  x1 = x0 + dt·(v0 + dt·a)
        let a = g; // constant acceleration = gravity for a free particle
        let v0 = Vector::new(1.0, 0.0, 0.0);
        let expected = Vector::new(
            x0.x + dt * v0.x + dt * dt * a.x,
            x0.y + dt * v0.y + dt * dt * a.y,
            x0.z + dt * v0.z + dt * dt * a.z,
        );
        assert_eq!(body.particles[0].pos.x.to_bits(), expected.x.to_bits());
        assert_eq!(body.particles[0].pos.y.to_bits(), expected.y.to_bits());
        assert_eq!(body.particles[0].pos.z.to_bits(), expected.z.to_bits());
        // And the velocity update is exact too.
        let expected_v = Vector::new(v0.x + dt * a.x, v0.y + dt * a.y, v0.z + dt * a.z);
        assert_eq!(body.particles[0].vel.x.to_bits(), expected_v.x.to_bits());
        assert_eq!(body.particles[0].vel.y.to_bits(), expected_v.y.to_bits());
        assert_eq!(body.particles[0].vel.z.to_bits(), expected_v.z.to_bits());
    }

    #[test]
    fn pinned_particle_does_not_move() {
        let mut body = SoftBody::new(Vector::new(0.0, -9.81, 0.0));
        let p = body.add_pinned(Vector::new(0.0, 5.0, 0.0));
        body.add_particle(Vector::new(0.0, 0.0, 0.0));
        // Attach a spring so forces are exercised; the pinned endpoint must stay put.
        body.add_spring(p, 1, 100.0, 1.0);

        let before = body.particles[p].pos;
        for _ in 0..50 {
            body.step(0.01);
        }
        assert_eq!(body.particles[p].pos.x.to_bits(), before.x.to_bits());
        assert_eq!(body.particles[p].pos.y.to_bits(), before.y.to_bits());
        assert_eq!(body.particles[p].pos.z.to_bits(), before.z.to_bits());
        assert_eq!(body.particles[p].inv_mass, 0.0);
    }

    #[test]
    fn spring_pulls_particles_together() {
        // Two free particles, spring stretched beyond rest, no gravity: they should
        // move toward each other (distance decreases), and energy must stay finite.
        let mut body = SoftBody::new(Vector::ZERO);
        let a = body.add_particle(Vector::new(-2.0, 0.0, 0.0));
        let b = body.add_particle(Vector::new(2.0, 0.0, 0.0));
        // rest length auto-set to current distance 4.0; shrink effective rest by
        // creating a second stiffer spring is overkill — instead verify a stretched
        // spring from a short rest pulls them in.
        body.springs.clear();
        body.springs.push(Spring {
            a,
            b,
            rest_length: 1.0,
            stiffness: 50.0,
            damping: 0.5,
        });

        let d0 = (body.particles[b].pos - body.particles[a].pos).length();
        for _ in 0..200 {
            body.step(0.005);
        }
        let d1 = (body.particles[b].pos - body.particles[a].pos).length();
        assert!(d1 < d0, "spring should shorten the gap: {d0} -> {d1}");
        assert!(body.kinetic_energy().is_finite());
        assert!(body.particles[a].pos.is_finite());
        assert!(body.particles[b].pos.is_finite());
    }

    #[test]
    fn write_spring_forces_routes_into_force_containers() {
        use crate::dynamics::{
            RigidBodyBuilder, RigidBodySet, RigidBodyType,
            force_containers::{ForceContainer, ForceKind},
        };

        // Two rigid bodies 4 units apart on the x-axis; bind one particle to each.
        let mut bodies = RigidBodySet::new();
        let builder_a = RigidBodyBuilder::new(RigidBodyType::Dynamic)
            .translation(Vector::new(-2.0, 0.0, 0.0).into());
        let builder_b = RigidBodyBuilder::new(RigidBodyType::Dynamic)
            .translation(Vector::new(2.0, 0.0, 0.0).into());
        let ha = bodies.insert(builder_a.build());
        let hb = bodies.insert(builder_b.build());

        // Soft body: two bound particles, spring rest = 1.0 (so the 4.0 gap pulls in).
        let mut sb = SoftBody::new(Vector::ZERO);
        let pa = sb.add_particle(Vector::new(-2.0, 0.0, 0.0));
        let pb = sb.add_particle(Vector::new(2.0, 0.0, 0.0));
        sb.particles[pa].bound_body = Some(ha);
        sb.particles[pb].bound_body = Some(hb);
        sb.add_spring(pa, pb, 50.0, 0.5); // rest auto-set to 4.0 by add_spring...

        // Override rest length to 1.0 so the stretched spring produces a known force.
        sb.springs[0].rest_length = 1.0;

        let mut set = SoftBodySet::new();
        let id = set.insert(sb);
        let _ = id;

        set.write_spring_forces(&mut bodies);

        let kind = ForceKind::Custom(SOFT_SPRING_CUSTOM_ID);
        let ca = bodies
            .get(ha)
            .unwrap()
            .force_containers
            .get(&kind)
            .expect("soft-spring container present on body A");
        let cb = bodies
            .get(hb)
            .unwrap()
            .force_containers
            .get(&kind)
            .expect("soft-spring container present on body B");

        // Read back the routed forces via the public contribution iterator.
        let fa = ca.contributions().next().unwrap().force();
        let fb = cb.contributions().next().unwrap().force();

        // Spring stretched: len=4, rest=1 → f_spring = 50*(4-1) = 150, along +x from A.
        assert!((fb.x + 150.0).abs() < 1e-9, "B force.x = {}", fb.x);
        // Equal and opposite, no y/z component.
        assert!(fa.y.abs() < 1e-12 && fa.z.abs() < 1e-12);
        assert!(fb.y.abs() < 1e-12 && fb.z.abs() < 1e-12);
    }
}
