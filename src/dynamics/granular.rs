//! Granular-body (DEM — Discrete Element Method) support — Phase 0.
//!
//! Sister module to `fluid.rs` (Phase 35): the same particle-cloud scaffolding
//! (`GranularParticle` / `GranularWorld` / `GranularParams`, naive O(n²)
//! neighbour search, semi-implicit Euler, no SoA boundary touched, fork-local
//! `f64` math), with the fluid's pressure/viscosity kernels replaced by a
//! granular contact model:
//!
//! * **Radial repulsion** — linear-spring push when two particles overlap
//!   (`d < r_i + r_j`), stiffness `k_n`, with velocity-proportional normal
//!   damping `c_n`. Hertzian-style: no attraction, no tension.
//! * **Tangential friction** — Coulomb-limited tangential damping on the
//!   relative tangential velocity, clamped to `μ · |F_n|`. This is the
//!   defining granular term: it lets a pile keep its slope (angle of repose)
//!   instead of flowing away like sand-coloured water.
//! * **Rolling resistance** — optional extra damping of the relative
//!   tangential velocity below the Coulomb cap (stabilises heaps; keeps the
//!   model purely force-based, no angular DOF in Phase 0).
//!
//! Determinism: fixed particle-index iteration order and `f64` only, so runs
//! are bit-identical under `enhanced-determinism` (same contract as
//! `FluidWorld::step` / `SoftBody`).
//!
//! Phase 0 = data structures + contact force model + integrator + tests.
//! Later phases wire `GranularWorld` into `PhysicsWorld` and add rigid-body /
//! voxel coupling (dig terrain → spawn grains).

use crate::alloc_prelude::Vec;
use crate::math::{Real, Vector};
use std::collections::HashMap;

/// A single DEM granular particle.
#[derive(Clone, Debug)]
pub struct GranularParticle {
    /// Current world-space position.
    pub pos: Vector,
    /// Current world-space linear velocity.
    pub vel: Vector,
    /// Accumulated acceleration for the current step (cleared each `step`).
    pub accel: Vector,
    /// Particle mass (> 0).
    pub mass: Real,
    /// Contact radius (> 0). Two particles touch when their centres are
    /// closer than the sum of their radii.
    pub radius: Real,
}

impl GranularParticle {
    /// Creates a free granular particle.
    pub fn new(pos: Vector, vel: Vector, mass: Real, radius: Real) -> Self {
        Self {
            pos,
            vel,
            accel: Vector::ZERO,
            mass,
            radius,
        }
    }
}

/// DEM tunable parameters for a [`GranularWorld`].
#[derive(Clone, Copy, Debug)]
pub struct GranularParams {
    /// Normal contact stiffness `k_n` (linear spring, N/m). Must be `> 0`.
    /// Keep `k_n / m · dt² < 1` for explicit-integrator stability.
    pub normal_stiffness: Real,
    /// Normal contact damping `c_n` (N·s/m), velocity-proportional along the
    /// contact normal. `>= 0`.
    pub normal_damping: Real,
    /// Coulomb friction coefficient `μ`. Tangential force is clamped to
    /// `μ · |F_n|`. `>= 0`.
    pub friction: Real,
    /// Tangential damping fraction in `[0, 1]`: scales the relative
    /// tangential velocity before the Coulomb clamp (rolling-resistance
    /// proxy). `0` = pure Coulomb cap on an undamped slide.
    pub tangential_damping: Real,
    /// Constant body acceleration (typically gravity).
    pub gravity: Vector,
}

impl Default for GranularParams {
    fn default() -> Self {
        Self {
            normal_stiffness: 800.0,
            normal_damping: 0.5,
            friction: 0.6,
            tangential_damping: 0.4,
            gravity: Vector::ZERO,
        }
    }
}

/// A cloud of [`GranularParticle`]s sharing one set of [`GranularParams`].
#[derive(Clone, Debug)]
pub struct GranularWorld {
    /// Particles. Index into this `Vec` is the particle id.
    pub particles: Vec<GranularParticle>,
    /// Shared DEM parameters.
    pub params: GranularParams,
}

impl GranularWorld {
    /// Creates an empty granular world with the given parameters.
    pub fn new(params: GranularParams) -> Self {
        Self {
            particles: Vec::new(),
            params,
        }
    }

    /// Number of particles.
    pub fn len(&self) -> usize {
        self.particles.len()
    }

    /// True when there are no particles.
    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }

    /// Appends a particle, returning its index.
    pub fn add_particle(&mut self, pos: Vector, vel: Vector, mass: Real, radius: Real) -> usize {
        let i = self.particles.len();
        self.particles
            .push(GranularParticle::new(pos, vel, mass, radius));
        i
    }

    /// Advance the granular cloud by one timestep `dt` using semi-implicit
    /// Euler.
    ///
    /// Per step: (1) accumulate pairwise contact forces (radial spring-damper
    /// + Coulomb-clamped tangential damping) over all i<j pairs; (2) add
    /// gravity; (3) integrate velocities then positions.
    pub fn step(&mut self, dt: Real) {
        let n = self.particles.len();
        if n == 0 {
            return;
        }
        let p = self.params;

        // (1) Pairwise contact forces via a uniform spatial hash broad-phase
        // (Phase 38): cell size = 2·r_max guarantees every touching pair lands
        // in adjacent cells, so scanning the 27 neighbours of each particle
        // finds all contacts. Determinism: buckets are filled in particle
        // order and each unordered pair is handled exactly once (j > i), so
        // the force sequence is reproducible run-to-run.
        let mut r_max: Real = 0.0;
        for part in &self.particles {
            r_max = r_max.max(part.radius);
        }
        let cell = 2.0 * r_max;
        let mut cells: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
        for (idx, part) in self.particles.iter().enumerate() {
            let key = (
                (part.pos.x / cell).floor() as i64,
                (part.pos.y / cell).floor() as i64,
                (part.pos.z / cell).floor() as i64,
            );
            cells.entry(key).or_default().push(idx);
        }
        let mut accels: Vec<Vector> = Vec::new();
        accels.resize(n, Vector::ZERO);
        for i in 0..n {
            let pi = self.particles[i].pos;
            let vi = self.particles[i].vel;
            let ri = self.particles[i].radius;
            let cx = (pi.x / cell).floor() as i64;
            let cy = (pi.y / cell).floor() as i64;
            let cz = (pi.z / cell).floor() as i64;
            for dx in -1..=1i64 {
                for dy in -1..=1i64 {
                    for dz in -1..=1i64 {
                        let Some(bucket) = cells.get(&(cx + dx, cy + dy, cz + dz)) else {
                            continue;
                        };
                        for &j in bucket {
                            if j <= i {
                                continue; // each unordered pair once
                            }
                            let pj = self.particles[j].pos;
                            let rij = pi - pj; // from j to i
                            let d2 = rij.length_squared();
                            let rsum = ri + self.particles[j].radius;
                            if d2 >= rsum * rsum {
                                continue; // not touching
                            }
                            let d = d2.max(1e-18).sqrt();
                            let n_hat = rij / d; // from j toward i
                            let overlap = rsum - d;
                            let vij = vi - self.particles[j].vel;
                            let v_n = vij.dot(n_hat); // approaching when negative

                            // Normal force on i: spring push apart + damping of approach.
                            let f_n_mag =
                                (p.normal_stiffness * overlap - p.normal_damping * v_n).max(0.0);
                            let f_n = n_hat * f_n_mag;

                            // Tangential relative velocity (slide direction).
                            let v_t_vec = vij - n_hat * v_n;
                            let v_t = v_t_vec.length();
                            // Tangential force on i opposes the slide, capped by μ·|F_n|.
                            let f_t = if v_t > 1e-12 {
                                let t_hat = v_t_vec / v_t;
                                let raw = p.tangential_damping * v_t;
                                let cap = p.friction * f_n_mag;
                                -t_hat * raw.min(cap)
                            } else {
                                Vector::ZERO
                            };

                            let f = f_n + f_t;
                            accels[i] += f / self.particles[i].mass;
                            accels[j] -= f / self.particles[j].mass;
                        }
                    }
                }
            }
        }

        // (2) Gravity + (3) semi-implicit Euler: velocities first, then
        // positions — same operation order as `FluidWorld::step`.
        for (i, part) in self.particles.iter_mut().enumerate() {
            let a = accels[i] + p.gravity;
            part.vel += a * dt;
            part.pos += part.vel * dt;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> GranularParams {
        GranularParams {
            normal_stiffness: 800.0,
            normal_damping: 0.5,
            friction: 0.0,
            tangential_damping: 0.0,
            gravity: Vector::new(0.0, -9.81, 0.0),
        }
    }

    #[test]
    fn dem_single_particle_free_fall_is_analytic() {
        // One particle, no neighbours → no contacts, pure gravity.
        let mut gw = GranularWorld::new(params());
        gw.add_particle(Vector::new(0.0, 0.0, 0.0), Vector::ZERO, 1.0, 0.05);
        let dt = 1.0 / 60.0;
        for _ in 0..60 {
            gw.step(dt);
        }
        let p = &gw.particles[0];
        // Semi-implicit Euler under constant gravity: x(t) = ½ g t² + ½ g·dt·t
        // (an O(dt) drift vs the analytic ½ g t²). With dt = 1/60 and t = 1 s
        // the drift is ≈ 0.082, so assert within 0.1.
        let expected = 0.5 * (-9.81) * 1.0_f64 * 1.0;
        assert!(
            (p.pos.y - expected).abs() < 0.1,
            "free-fall y={} expected≈{}",
            p.pos.y,
            expected
        );
        assert!(p.pos.y < -4.0, "particle fell downward under gravity");
        assert!(p.pos.x.abs() < 1e-12 && p.pos.z.abs() < 1e-12);
    }

    #[test]
    fn dem_overlapping_particles_push_apart() {
        // Two particles overlapping → spring pushes them apart until they
        // separate; centre of mass stays put (symmetric forces).
        let mut p = params();
        p.normal_damping = 2.0; // settle the bounce
        p.gravity = Vector::ZERO; // isolate the contact response
        let mut gw = GranularWorld::new(p);
        gw.add_particle(Vector::new(-0.03, 0.0, 0.0), Vector::ZERO, 1.0, 0.05);
        gw.add_particle(Vector::new(0.03, 0.0, 0.0), Vector::ZERO, 1.0, 0.05);
        let dt = 1.0 / 120.0;
        for _ in 0..240 {
            gw.step(dt);
        }
        let sep = (gw.particles[1].pos - gw.particles[0].pos).length();
        assert!(
            sep >= 0.1 - 1e-3,
            "overlapping particles should push apart to ~sum of radii, sep={sep}"
        );
        let com = (gw.particles[0].pos + gw.particles[1].pos) / 2.0;
        assert!(
            com.x.abs() < 1e-9 && com.y.abs() < 1e-6 && com.z.abs() < 1e-9,
            "centre of mass should stay put, com={com:?}"
        );
    }

    #[test]
    fn dem_friction_damps_tangential_slide() {
        // Two overlapping particles, relative velocity purely tangential.
        // The lower one is frozen (huge mass); the upper one slides over it.
        // With μ > 0 the tangential slide must decay during the brief contact
        // (the Coulomb clamp allows a friction force); with μ = 0 it cannot.
        let slide_after = |friction: Real| {
            let mut p = params();
            p.friction = friction;
            p.tangential_damping = 1.0;
            p.normal_damping = 5.0;
            // Soft normal spring → the contact persists long enough for the
            // tangential friction to accumulate a visible impulse.
            p.normal_stiffness = 100.0;
            p.gravity = Vector::ZERO;
            let mut gw = GranularWorld::new(p);
            gw.add_particle(Vector::new(0.0, 0.0, 0.0), Vector::ZERO, 1.0e9, 0.5);
            // Overlap 0.04 along x; slide velocity 2 m/s along y (tangential).
            gw.add_particle(
                Vector::new(0.96, 0.0, 0.0),
                Vector::new(0.0, 2.0, 0.0),
                1.0,
                0.5,
            );
            let dt = 1.0 / 240.0;
            for _ in 0..40 {
                gw.step(dt);
            }
            gw.particles[1].vel.y
        };
        let v_frictionless = slide_after(0.0);
        let v_friction = slide_after(0.8);
        assert!(
            (v_frictionless - 2.0).abs() < 0.15,
            "without friction the slide is essentially untouched: v={v_frictionless}"
        );
        assert!(
            v_friction < v_frictionless - 0.1,
            "friction should slow the slide: μ=0.8 v={v_friction} vs μ=0 v={v_frictionless}"
        );
    }

    #[test]
    fn dem_step_is_deterministic() {
        let mut a = GranularWorld::new(params());
        a.add_particle(Vector::new(0.0, 0.0, 0.0), Vector::ZERO, 1.0, 0.05);
        a.add_particle(Vector::new(0.08, 0.01, 0.0), Vector::ZERO, 1.0, 0.05);
        a.add_particle(Vector::new(-0.06, 0.02, 0.05), Vector::ZERO, 1.0, 0.05);
        let mut b = a.clone();
        let dt = 1.0 / 60.0;
        for _ in 0..20 {
            a.step(dt);
            b.step(dt);
        }
        for (pa, pb) in a.particles.iter().zip(b.particles.iter()) {
            assert_eq!(pa.pos, pb.pos, "positions bit-identical across runs");
            assert_eq!(pa.vel, pb.vel, "velocities bit-identical across runs");
        }
    }
}
