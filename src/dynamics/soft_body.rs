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

use crate::math::{Real, Vector};
use std::vec::Vec;

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
}

impl SoftParticle {
    /// Creates a free (movable) particle with unit mass.
    pub fn new(pos: Vector) -> Self {
        Self {
            pos,
            vel: Vector::ZERO,
            force: Vector::ZERO,
            inv_mass: 1.0,
        }
    }

    /// Creates a pinned (immovable) particle — its `inv_mass` is `0`.
    pub fn pinned(pos: Vector) -> Self {
        Self {
            pos,
            vel: Vector::ZERO,
            force: Vector::ZERO,
            inv_mass: 0.0,
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
}

impl SoftBody {
    /// Creates an empty soft body in a gravity field `gravity`.
    pub fn new(gravity: Vector) -> Self {
        Self {
            particles: Vec::new(),
            springs: Vec::new(),
            gravity,
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

    /// Accumulates the total force on every particle: gravity plus all spring
    /// (Hookean + axial damping) contributions. Clears each particle's `force`
    /// first, so this can be called once per substep before `integrate`.
    pub fn compute_forces(&mut self) {
        for p in &mut self.particles {
            p.force = if p.inv_mass == 0.0 {
                Vector::ZERO
            } else {
                p.mass() * self.gravity
            };
        }

        for s in &self.springs {
            // Copy the endpoints' data by value so the immutable borrow of
            // `self.particles` ends before we mutate the forces below.
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
            // Unit axis from `a` to `b`.
            let dir = delta / len;
            // Hookean term: pull the pair toward `rest_length`.
            let f_spring = s.stiffness * (len - s.rest_length);
            // Axial damping: oppose relative velocity along the axis.
            let rel_vel = pb_vel - pa_vel;
            let f_damp = s.damping * rel_vel.dot(dir);
            // Total axial force magnitude (positive = stretching).
            let f_axial = f_spring + f_damp;
            let f = dir * f_axial;

            // Apply equal and opposite forces (skip pinned endpoints: inv_mass == 0).
            if pa_im != 0.0 {
                self.particles[s.a].force += f;
            }
            if pb_im != 0.0 {
                self.particles[s.b].force -= f;
            }
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

    /// Advances every soft body by `dt`.
    pub fn step(&mut self, dt: Real) {
        for body in &mut self.bodies {
            body.step(dt);
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
}
