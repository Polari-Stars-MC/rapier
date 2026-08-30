//! Fluid-body (SPH — Smoothed-Particle Hydrodynamics) support — Phase 0.
//!
//! This module is the **data-structure + particle integrator** skeleton for
//! incompressible-fluid simulation, on par with `soft_body.rs` (Phase 0). It is
//! intentionally independent of the SoA SIMD solver boundary so it can be
//! compiled and unit-tested in isolation.
//!
//! ## Design notes (see `.hermes/plans/2026-08-30_fluid-sph-roadmap.md`)
//!
//! * A fluid is a cloud of `FluidParticle`s. Each carries position, velocity,
//!   accumulated force, mass, and (per-step) density / pressure.
//! * Integration is **semi-implicit (symplectic) Euler**: velocities first
//!   (`v += dt·a`), then positions (`x += dt·v`) — the same operation order used
//!   by `SoftBody::integrate` and the rigid-body integrator, keeping the
//!   floating-point sequence bit-identical across runs under
//!   `enhanced-determinism`.
//! * SPH kernels (Poly6 density, Spiky pressure gradient, viscosity Laplacian)
//!   are implemented locally in `f64` (reusing `crate::math::Vector`/`Real`),
//!   so the fork stays self-contained — it does **not** depend on `mps-formula`
//!   (which keeps its own SPH formula layer for the mps-core estimation FFI).
//! * Neighbour search is naive O(n²) in Phase 0 (small scale, bit-identical);
//!   a spatial hash arrives in a later phase.
//!
//! Phase 0 = data structures + SPH force model + integrator + tests. Later
//! phases wire `FluidWorld` into `PhysicsWorld` and add rigid-body coupling.

use crate::alloc_prelude::Vec;
use crate::math::{Real, Vector};

/// A single SPH fluid particle.
#[derive(Clone, Debug)]
pub struct FluidParticle {
    /// Current world-space position.
    pub pos: Vector,
    /// Current world-space linear velocity.
    pub vel: Vector,
    /// Accumulated acceleration for the current step (cleared each `step`).
    pub accel: Vector,
    /// Particle mass (> 0).
    pub mass: Real,
    /// Recomputed each step: local SPH density (Poly6 sum of neighbour masses).
    pub density: Real,
    /// Recomputed each step: pressure `max(gas·(density − rest_density), 0)`.
    pub pressure: Real,
}

impl FluidParticle {
    /// Creates a free fluid particle.
    pub fn new(pos: Vector, vel: Vector, mass: Real) -> Self {
        Self {
            pos,
            vel,
            accel: Vector::ZERO,
            mass,
            density: 0.0,
            pressure: 0.0,
        }
    }
}

/// SPH tunable parameters for a [`FluidWorld`].
#[derive(Clone, Copy, Debug)]
pub struct FluidParams {
    /// Smoothing radius `h` — kernel cutoff. Particles farther than `h` do not
    /// interact. Must be `> 0`.
    pub smoothing_radius: Real,
    /// Equation-of-state gas constant `k` (Tait/Murnaghan stiffness).
    pub gas_constant: Real,
    /// Rest density `ρ₀` (target density at rest). Must be `> 0`.
    pub rest_density: Real,
    /// Dynamic viscosity `μ` (velocity-diffusion / cohesion). `>= 0`.
    pub viscosity: Real,
    /// Surface tension coefficient `σ` (optional, Phase 0 keeps it for API
    /// completeness; the force model uses it as an extra inward pull toward the
    /// local centroid of neighbours). `>= 0`.
    pub surface_tension: Real,
    /// Constant body acceleration (typically gravity).
    pub gravity: Vector,
}

impl Default for FluidParams {
    fn default() -> Self {
        Self {
            smoothing_radius: 1.0,
            gas_constant: 100.0,
            rest_density: 1000.0,
            viscosity: 0.1,
            surface_tension: 0.0,
            gravity: Vector::ZERO,
        }
    }
}

/// A cloud of `FluidParticle`s sharing one set of [`FluidParams`].
#[derive(Clone, Debug)]
pub struct FluidWorld {
    /// Particles. Index into this `Vec` is the particle id.
    pub particles: Vec<FluidParticle>,
    /// Shared SPH parameters.
    pub params: FluidParams,
}

impl FluidWorld {
    /// Creates an empty fluid world with the given parameters.
    pub fn new(params: FluidParams) -> Self {
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
    pub fn add_particle(&mut self, pos: Vector, vel: Vector, mass: Real) -> usize {
        let i = self.particles.len();
        self.particles.push(FluidParticle::new(pos, vel, mass));
        i
    }

    /// SPH Poly6 kernel `W(r, h)` — used for density estimation.
    fn poly6(distance: Real, h: Real) -> Real {
        if distance >= h {
            return 0.0;
        }
        let h2 = h * h;
        let r2 = distance * distance;
        let diff = h2 - r2;
        if diff <= 0.0 {
            return 0.0;
        }
        315.0 / (64.0 * std::f64::consts::PI * h.powi(9)) * diff.powi(3)
    }

    /// SPH Spiky pressure-gradient kernel `∇W(r, h)` (points from neighbour to
    /// self, i.e. along `-r̂` scaled by the spiky slope). Returns the zero vector
    /// when `distance <= 0` or `>= h`.
    fn spiky_gradient(offset: Vector, h: Real) -> Vector {
        let distance = offset.length();
        if distance <= 1e-9 || distance >= h {
            return Vector::ZERO;
        }
        let diff = h - distance;
        // -r̂ · (45 / (π h⁶)) · (h - r)²  — standard spiky gradient.
        -offset / distance * (45.0 / (std::f64::consts::PI * h.powi(6)) * diff * diff)
    }

    /// SPH viscosity Laplacian `∇²W(r, h)`.
    fn viscosity_laplacian(distance: Real, h: Real) -> Real {
        if distance >= h {
            return 0.0;
        }
        45.0 / (std::f64::consts::PI * h.powi(6)) * (h - distance)
    }

    /// Advance the fluid by one timestep `dt` using semi-implicit Euler.
    ///
    /// Per step: (1) recompute each particle's density + pressure from its
    /// neighbours; (2) compute the SPH pressure + viscosity + gravity
    /// acceleration; (3) integrate velocities then positions.
    pub fn step(&mut self, dt: Real) {
        let n = self.particles.len();
        if n == 0 {
            return;
        }
        let h = self.params.smoothing_radius;
        let h2 = h * h;
        let k = self.params.gas_constant;
        let rho0 = self.params.rest_density;
        let mu = self.params.viscosity;

        // (1) Density + pressure for every particle.
        for i in 0..n {
            let pi = self.particles[i].pos;
            let mi = self.particles[i].mass;
            let mut density = Self::poly6(0.0, h) * mi; // self contribution
            for j in 0..n {
                let pj = self.particles[j].pos;
                let mj = self.particles[j].mass;
                let d2 = (pi - pj).length_squared();
                if d2 < h2 {
                    density += mj * Self::poly6(d2.sqrt(), h);
                }
            }
            let density = density.max(1e-9);
            let pressure = (k * (density - rho0)).max(0.0);
            self.particles[i].density = density;
            self.particles[i].pressure = pressure;
        }

        // (2) Accelerations.
        let mut accels: Vec<Vector> = Vec::new();
        accels.resize(n, Vector::ZERO);
        for i in 0..n {
            let pi = self.particles[i].pos;
            let vi = self.particles[i].vel;
            let rho_i = self.particles[i].density;
            let p_i = self.particles[i].pressure;
            let mut pressure_force = Vector::ZERO;
            let mut viscosity_force = Vector::ZERO;
            let mut centroid = Vector::ZERO;
            let mut nbr = 0_usize;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let pj = self.particles[j].pos;
                let vj = self.particles[j].vel;
                let rho_j = self.particles[j].density;
                let p_j = self.particles[j].pressure;
                let rij = pi - pj; // from j to i
                let d2 = rij.length_squared();
                if d2 >= h2 || d2 <= 1e-18 {
                    continue;
                }
                let d = d2.sqrt();
                // Pressure force (symmetrised): -Σ m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W_ij
                let grad = Self::spiky_gradient(rij, h);
                pressure_force +=
                    self.particles[j].mass * (p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j)) * grad;
                // Viscosity force: μ Σ m_j (v_j − v_i)/ρ_j ∇²W_ij
                let lap = Self::viscosity_laplacian(d, h);
                viscosity_force += self.particles[j].mass * (vj - vi) / rho_j * lap;
                centroid += pj;
                nbr += 1;
            }
            let pressure_accel = -pressure_force; // force already carries the -grad sign
            let viscosity_accel = mu * viscosity_force;
            let mut a = pressure_accel + viscosity_accel + self.params.gravity;
            if self.params.surface_tension > 0.0 && nbr > 0 {
                centroid /= nbr as Real;
                // Pull toward neighbour centroid (cohesion).
                a += self.params.surface_tension * (centroid - pi);
            }
            accels[i] = a;
        }

        // (3) Semi-implicit Euler.
        for i in 0..n {
            let a = accels[i];
            let p = &mut self.particles[i];
            p.vel += a * dt;
            p.pos += p.vel * dt;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> FluidParams {
        FluidParams {
            smoothing_radius: 1.0,
            gas_constant: 100.0,
            rest_density: 1000.0,
            viscosity: 0.0,
            surface_tension: 0.0,
            gravity: Vector::new(0.0, -9.81, 0.0),
        }
    }

    #[test]
    fn sph_single_particle_free_fall_is_analytic() {
        // One particle, no neighbours → no pressure/viscosity, pure gravity.
        let mut fw = FluidWorld::new(params());
        fw.add_particle(Vector::new(0.0, 0.0, 0.0), Vector::ZERO, 1.0);
        let dt = 1.0 / 60.0;
        for _ in 0..60 {
            fw.step(dt);
        }
        let p = &fw.particles[0];
        // Semi-implicit Euler under constant gravity: x(t) = ½ g t² + ½ g·dt·t
        // (an O(dt) drift vs the analytic ½ g t²). With dt = 1/60 and t = 1 s the
        // drift is ≈ 0.082, so we assert within 0.1 and that it falls monotonically.
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
    fn sph_two_particles_repel_under_compression() {
        // Two particles closer than rest spacing → local density > rest_density
        // → positive pressure → they push apart (centre of mass stays fixed).
        // rest_density is set low (2.0) so the ~3.1 local density of two unit-mass
        // particles 0.1 apart exceeds it and yields positive pressure.
        let mut p = params();
        p.rest_density = 2.0;
        p.gas_constant = 200.0;
        p.viscosity = 0.0;
        let mut fw = FluidWorld::new(p);
        fw.add_particle(Vector::new(-0.05, 0.0, 0.0), Vector::ZERO, 1.0);
        fw.add_particle(Vector::new(0.05, 0.0, 0.0), Vector::ZERO, 1.0);
        let dt = 1.0 / 120.0;
        for _ in 0..60 {
            fw.step(dt);
        }
        let sep = (fw.particles[1].pos - fw.particles[0].pos).length();
        assert!(
            sep > 0.1,
            "compressed particles should repel, sep={sep} (started 0.1)"
        );
        // Centre of mass x/z stay at origin (symmetric start, symmetric forces).
        // y is free to fall under gravity (both particles drop together).
        let com = (fw.particles[0].pos + fw.particles[1].pos) / 2.0;
        assert!(
            com.x.abs() < 1e-9 && com.z.abs() < 1e-9,
            "centre of mass x/z should stay at origin, com={com:?}"
        );
    }

    #[test]
    fn sph_step_is_deterministic() {
        let mut a = FluidWorld::new(params());
        a.add_particle(Vector::new(0.0, 0.0, 0.0), Vector::ZERO, 1.0);
        a.add_particle(Vector::new(0.2, 0.1, 0.0), Vector::ZERO, 1.0);
        a.add_particle(Vector::new(-0.15, 0.05, 0.1), Vector::ZERO, 1.0);
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
