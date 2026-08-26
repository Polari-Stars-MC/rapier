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
use std::collections::HashSet;
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
    /// Phase 8: when `bound_body` is `Some`, this is the attachment point
    /// expressed in the bound body's *local* frame (so the particle rigidly
    /// follows the body as it translates/rotates). Computed at attach time from
    /// the world-space attach point; ignored when `bound_body` is `None`.
    pub bound_local: Vector,
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
            bound_local: Vector::ZERO,
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
            bound_local: Vector::ZERO,
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

/// A distance constraint (XPBD) between two particles — the edge of a deformable
/// mesh. Unlike [`Spring`] (a force), this is a *position* constraint solved by
/// XPBD's small-iteration projection, which is unconditionally stable even for
/// stiff/rigid edges (no explicit-spring explosion).
#[derive(Clone, Copy, Debug)]
pub struct DistanceConstraint {
    /// First endpoint particle index.
    pub a: usize,
    /// Second endpoint particle index.
    pub b: usize,
    /// Rest length.
    pub rest: Real,
    /// XPBD compliance `α` (0 = rigid, > 0 = soft). Stored per-constraint so
    /// different edges can have different stiffness.
    pub compliance: Real,
}

/// Which integrator a [`SoftBody`] uses.
///
/// * `MassSpring` — the Phase 0/2 Hookean-spring + semi-implicit Euler path.
/// * `Xpbd { iterations, compliance }` — Phase 3 position-based dynamics: edges
///   become [`DistanceConstraint`]s and tetrahedra carry a volume constraint,
///   both projected with a fixed number of Gauss-Seidel iterations. `compliance`
///   is the default used when a constraint is added without an explicit one.
#[derive(Clone, Copy, Debug, Default)]
pub enum SoftSolver {
    /// Hookean springs (default, Phase 0/2).
    #[default]
    MassSpring,
    /// XPBD position-based solver (Phase 3).
    Xpbd {
        /// Gauss-Seidel projection iterations per substep.
        iterations: u32,
        /// Default XPBD compliance for constraints created without an explicit one.
        compliance: Real,
    },
}

/// A uniform wind / air-resistance field applied to every free particle of a
/// soft body (Phase 7). It is a *pure external force* — no new mechanics, it
/// reuses the same force path as gravity:
///
/// * `accel` — constant wind acceleration (a directional push, like a sideways
///   gravity). Makes a pinned-edge cloth fly out like a flag.
/// * `drag` — linear air-resistance coefficient. Each free particle feels
///   `F = m·accel − m·drag·v`, i.e. a velocity damping toward the wind state.
///   Keep `drag·dt < 1` for stability (the integration clamp handles it too).
#[derive(Clone, Copy, Debug)]
pub struct Wind {
    /// Constant wind acceleration (`m/s²`), applied to every free particle.
    pub accel: Vector,
    /// Linear air-resistance coefficient (`1/s`). `F_drag = -m·drag·v`.
    pub drag: Real,
}

/// A soft body: a collection of point masses connected by springs.
///
/// In Phase 0/2 it is a mass-spring cloud. Phase 3 adds an optional XPBD
/// position-based solver: edges become [`DistanceConstraint`]s and tetrahedra
/// carry a volume constraint, projected each substep. The `solver` field selects
/// which path [`SoftBody::step`] takes.
#[derive(Clone, Debug)]
pub struct SoftBody {
    /// Point masses. Spring endpoints / constraint indices index into this `Vec`.
    pub particles: Vec<SoftParticle>,
    /// Springs (edges) between particles — used by the `MassSpring` solver.
    pub springs: Vec<Spring>,
    /// Distance constraints (edges) — used by the `Xpbd` solver.
    pub distance_constraints: Vec<DistanceConstraint>,
    /// Tetrahedral volume elements. Each entry is `[a, b, c, d]` particle indices.
    /// Used by the `Xpbd` solver's volume-preservation constraint.
    pub tetrahedra: Vec<[u32; 4]>,
    /// Rest (reference) signed volume of each tetrahedron, precomputed at
    /// `add_tetrahedron` time. Indexed parallel to `tetrahedra`.
    pub tetra_rest_volumes: Vec<Real>,
    /// Triangular faces (cloth / shell topology). Each entry is `[a, b, c]`
    /// particle indices, CCW for outward normal. Phase 6: cloth soft bodies are
    /// built from triangles; the structural edges are added automatically as
    /// distance constraints (see `add_triangle`), so the XPBD solver needs no new
    /// mechanics — bending is just extra distance constraints between opposite
    /// vertices of adjacent quads (composed by the caller).
    pub triangles: Vec<[u32; 3]>,
    /// Active integrator.
    pub solver: SoftSolver,
    /// Constant acceleration applied to every free particle (typically gravity).
    pub gravity: Vector,
    /// Coarse sleeping flag (island-style). When `true`, [`SoftBody::step`] is a
    /// no-op: the whole body is treated as an inactive unit, mirroring how
    /// `PersistentIslands` keeps a sleeping rigid-body island from being
    /// re-simulated. Per-particle island membership is Phase 3; this flag gives
    /// the same "skip inactive work" behavior at body granularity for now.
    pub sleeping: bool,
    /// Phase 5f: collision coupling flag. When `true` the soft body's particles
    /// are driven by external proxy rigid bodies (one `Ball` collider per free
    /// particle, maintained by the mps-core integration layer), so [`SoftBody::step`]
    /// must NOT integrate the particles itself — their positions/velocities are
    /// written back from the proxy bodies after the rigid-body narrow-phase/contact
    /// step. Forces (springs + gravity) are still computed and exported by the
    /// integration layer. Defaults to `false`.
    pub collide: bool,
    /// Phase 5f: proxy collider radius used when `collide` is enabled. Each free
    /// particle gets a `Ball` collider of this radius. Defaults to `0.1`.
    pub particle_radius: Real,
    /// Phase 7: uniform wind / air-resistance field. `None` = no wind. When set,
    /// every free particle feels `F = m·wind.accel − m·wind.drag·v` in addition to
    /// gravity — a pure external force, no new solver mechanics. Applied in both
    /// the `MassSpring` (`compute_forces`) and `Xpbd` (`step_xpbd` predict) paths.
    pub wind: Option<Wind>,
    /// Phase 9: tearing threshold. When `Some(ε)`, any structural edge (XPBD
    /// distance constraint or MassSpring spring) whose strain `(|len| − rest)/rest`
    /// exceeds `ε` is removed at the start of each [`SoftBody::step`]. Triangular
    /// faces that lose any structural edge are dropped too, so a torn cloth stops
    /// rendering the broken face. `None` (default) = no tearing. Pure topology
    /// edit — no new solver mechanics, no SoA interaction.
    pub tear_strain: Option<Real>,
}

impl SoftBody {
    /// Creates an empty soft body in a gravity field `gravity`.
    pub fn new(gravity: Vector) -> Self {
        Self {
            particles: Vec::new(),
            springs: Vec::new(),
            distance_constraints: Vec::new(),
            tetrahedra: Vec::new(),
            tetra_rest_volumes: Vec::new(),
            triangles: Vec::new(),
            solver: SoftSolver::MassSpring,
            gravity,
            sleeping: false,
            collide: false,
            particle_radius: 0.1,
            wind: None,
            tear_strain: None,
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

    /// Phase 8: anchors `particle` to a rigid body `body`, so it rigidly follows
    /// that body's motion. `world_attach_point` is the world-space point where the
    /// particle binds (usually the particle's current position). The point is
    /// stored in the body's *local* frame so the particle tracks translation and
    /// rotation. The particle stops integrating locally; its spring/damping force
    /// is instead routed to the body via [`SoftBodySet::write_spring_forces`].
    ///
    /// Returns `false` if `particle` is out of range or `body` is not in `bodies`.
    pub fn attach_particle(
        &mut self,
        particle: usize,
        body: RigidBodyHandle,
        world_attach_point: Vector,
        bodies: &RigidBodySet,
    ) -> bool {
        let Some(rb) = bodies.get(body) else {
            return false;
        };
        let local = rb.position().inverse_transform_point(world_attach_point);
        let Some(p) = self.particles.get_mut(particle) else {
            return false;
        };
        p.bound_body = Some(body);
        p.bound_local = local;
        true
    }

    /// Phase 8: detaches `particle` from any bound rigid body (returns it to a
    /// free, locally-integrated particle). No-op if already free.
    pub fn detach_particle(&mut self, particle: usize) -> bool {
        let Some(p) = self.particles.get_mut(particle) else {
            return false;
        };
        p.bound_body = None;
        p.bound_local = Vector::ZERO;
        true
    }

    /// Phase 7: enables a uniform wind / air-resistance field for this body.
    /// `accel` is a constant wind acceleration applied to every free particle
    /// (like a sideways gravity); `drag` is a linear air-resistance coefficient
    /// (`F_drag = −m·drag·v`). See [`Wind`]. Pass `accel = ZERO, drag = 0` to
    /// get the same effect as [`Self::clear_wind`].
    pub fn apply_wind(&mut self, accel: Vector, drag: Real) {
        self.wind = Some(Wind { accel, drag });
    }

    /// Phase 7: disables the wind field (`None`).
    pub fn clear_wind(&mut self) {
        self.wind = None;
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
            // Phase 7: uniformly applied wind / air-resistance (pure external force).
            if let Some(wind) = self.wind {
                p.force += p.mass() * wind.accel;
                p.force -= p.mass() * wind.drag * p.vel;
            }
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

    /// One substep: integrate according to the active [`SoftSolver`].
    ///
    /// * `MassSpring` → [`Self::step_mass_spring`] (Hookean forces, semi-implicit Euler).
    /// * `Xpbd` → [`Self::step_xpbd`] (distance + volume constraints, position-based).
    pub fn step(&mut self, dt: Real) {
        if self.sleeping {
            return;
        }
        // Phase 9: remove over-stretched structural edges before integrating.
        // (No-op unless `tear_strain` is `Some`.)
        self.tear();
        // Phase 5f: when collision coupling is on, the integration layer drives
        // particle positions/velocities from proxy rigid bodies (after the
        // rigid-body narrow-phase/contact step), so we must not integrate here.
        if self.collide {
            return;
        }
        match self.solver {
            SoftSolver::MassSpring => self.step_mass_spring(dt),
            SoftSolver::Xpbd { .. } => self.step_xpbd(dt),
        }
    }

    /// Mass-spring substep (Phase 0/2): accumulate Hookean + damping forces, then
    /// integrate with semi-implicit Euler.
    pub fn step_mass_spring(&mut self, dt: Real) {
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

    // ── Phase 3: XPBD setup ────────────────────────────────────────────────

    /// Switches this body to the XPBD solver with the given iteration count and
    /// default compliance. Existing springs are ignored in XPBD mode; add distance
    /// constraints and tetrahedra instead.
    pub fn configure_xpbd(&mut self, iterations: u32, compliance: Real) {
        self.solver = SoftSolver::Xpbd {
            iterations,
            compliance,
        };
    }

    /// Adds a distance constraint between particles `a` and `b`. The rest length
    /// is taken from the current distance (0 is rejected to avoid a degenerate
    /// axis). Returns `None` if indices are out of bounds or coincide.
    pub fn add_distance_constraint(
        &mut self,
        a: usize,
        b: usize,
        compliance: Real,
    ) -> Option<usize> {
        let (pa, pb) = (self.particles.get(a)?, self.particles.get(b)?);
        let rest = (pb.pos - pa.pos).length();
        if rest == 0.0 {
            return None;
        }
        let idx = self.distance_constraints.len();
        self.distance_constraints.push(DistanceConstraint {
            a,
            b,
            rest,
            compliance,
        });
        Some(idx)
    }

    /// Adds a tetrahedral volume element `[a, b, c, d]`. The rest (reference)
    /// signed volume is computed from the current particle positions and cached.
    /// Returns `None` if any index is out of bounds or duplicated.
    pub fn add_tetrahedron(&mut self, tet: [u32; 4]) -> Option<usize> {
        let [a, b, c, d] = tet;
        let (pa, pb, pc, pd) = (
            self.particles.get(a as usize)?,
            self.particles.get(b as usize)?,
            self.particles.get(c as usize)?,
            self.particles.get(d as usize)?,
        );
        // Reject degenerate (duplicate) indices.
        if a == b || a == c || a == d || b == c || b == d || c == d {
            return None;
        }
        let vol = signed_tetra_volume(pa.pos, pb.pos, pc.pos, pd.pos);
        let idx = self.tetrahedra.len();
        self.tetrahedra.push(tet);
        self.tetra_rest_volumes.push(vol);
        Some(idx)
    }

    /// Phase 6 — cloth: adds a triangular face `[a, b, c]` (CCW for outward
    /// normal) to the body's shell topology **and** automatically registers its
    /// three structural edges as distance constraints (rest length from the
    /// current particle spacing) so the existing XPBD solver keeps the face
    /// shape. Duplicate edges (shared by neighbouring triangles) are silently
    /// de-duplicated against existing distance constraints to avoid double
    /// stiffness. Returns `None` (and does nothing) if any index is out of
    /// bounds or duplicated, or if the face is degenerate (a zero-length edge).
    ///
    /// Bending stiffness is *not* added here: it is composed by the caller via
    /// [`Self::add_distance_constraint`] between opposite vertices of adjacent
    /// quad pairs (cross-diagonal edges), which needs no new mechanics.
    pub fn add_triangle(&mut self, tri: [u32; 3]) -> Option<usize> {
        let [a, b, c] = tri;
        let (pa, pb, pc) = (
            self.particles.get(a as usize)?,
            self.particles.get(b as usize)?,
            self.particles.get(c as usize)?,
        );
        if a == b || a == c || b == c {
            return None;
        }
        // Reject degenerate faces (any edge has zero rest length).
        let ab = (pb.pos - pa.pos).length();
        let bc = (pc.pos - pb.pos).length();
        let ca = (pa.pos - pc.pos).length();
        if ab == 0.0 || bc == 0.0 || ca == 0.0 {
            return None;
        }
        // Register structural edges (a-b, b-c, c-a) as distance constraints,
        // skipping any edge already present (shared by a neighbour triangle).
        for (u, v, rest) in [(a, b, ab), (b, c, bc), (c, a, ca)] {
            let exists = self.distance_constraints.iter().any(|d| {
                (d.a == u as usize && d.b == v as usize) || (d.a == v as usize && d.b == u as usize)
            });
            if !exists {
                self.distance_constraints.push(DistanceConstraint {
                    a: u as usize,
                    b: v as usize,
                    rest,
                    // Default compliance; tune via configure_xpbd / explicit later.
                    compliance: 0.0,
                });
            }
        }
        let idx = self.triangles.len();
        self.triangles.push(tri);
        Some(idx)
    }

    /// Phase 6 — cloth bending: adds a single bending edge between two particles
    /// `p` and `q` as a distance constraint (rest length from current spacing).
    /// Compose bending across a quad by calling this for its two diagonals, or
    /// across a fold line by linking the un-shared vertices of two adjacent
    /// triangles. Reuses the existing XPBD distance solver (no new mechanics).
    /// Returns `None` (and does nothing) if indices are out of bounds or the
    /// endpoints coincide.
    pub fn add_bending_constraint(&mut self, p: usize, q: usize) -> Option<usize> {
        let (pp, pq) = (self.particles.get(p)?, self.particles.get(q)?);
        let rest = (pq.pos - pp.pos).length();
        if rest == 0.0 {
            return None;
        }
        let idx = self.distance_constraints.len();
        self.distance_constraints.push(DistanceConstraint {
            a: p,
            b: q,
            rest,
            compliance: 0.0,
        });
        Some(idx)
    }

    /// Phase 9: enables/disables tearing. `strain` is the max allowed strain
    /// `(|len| − rest)/rest` before a structural edge snaps. Pass `None` to
    /// disable (default). A value ≤ 0 tears immediately on any stretch.
    pub fn set_tear_strain(&mut self, strain: Option<Real>) {
        self.tear_strain = strain;
    }

    /// Phase 9: removes every over-stretched structural edge. Called once at the
    /// top of [`SoftBody::step`]; a no-op when `tear_strain` is `None`.
    ///
    /// Edge strain is `s = (|len| − rest) / rest`. An XPBD distance constraint or
    /// MassSpring spring with `s > threshold` is dropped. Triangular faces that
    /// lose any of their structural edges (a-b, b-c, c-a) are dropped as well, so
    /// a torn cloth stops rendering the broken face. Pure topology edit — neither
    /// the SoA solver nor the integration order is touched.
    pub fn tear(&mut self) {
        let threshold = match self.tear_strain {
            Some(t) => t,
            None => return,
        };
        if threshold <= 0.0 {
            return; // 0 / negative would tear everything on the first step.
        }

        // Collect the surviving distance-constraint edges (and their indices).
        let mut keep_dc: Vec<(usize, DistanceConstraint)> = Vec::new();
        for (i, c) in self.distance_constraints.iter().enumerate() {
            let (pa, pb) = match (self.particles.get(c.a), self.particles.get(c.b)) {
                (Some(a), Some(b)) => (a, b),
                _ => continue, // dangling edge → drop.
            };
            let len = (pb.pos - pa.pos).length();
            let strain = if c.rest > 0.0 {
                (len - c.rest) / c.rest
            } else {
                0.0
            };
            if strain <= threshold {
                keep_dc.push((i, *c));
            }
        }
        let broken_dc: HashSet<usize> = {
            let mut s = HashSet::new();
            for i in 0..self.distance_constraints.len() {
                if !keep_dc.iter().any(|(ki, _)| *ki == i) {
                    s.insert(i);
                }
            }
            s
        };
        self.distance_constraints = keep_dc.into_iter().map(|(_, c)| c).collect();

        // Same for MassSpring springs.
        let mut keep_sp: Vec<Spring> = Vec::new();
        for s in self.springs.drain(..) {
            let (pa, pb) = match (self.particles.get(s.a), self.particles.get(s.b)) {
                (Some(a), Some(b)) => (a, b),
                _ => continue,
            };
            let len = (pb.pos - pa.pos).length();
            let strain = if s.rest_length > 0.0 {
                (len - s.rest_length) / s.rest_length
            } else {
                0.0
            };
            if strain <= threshold {
                keep_sp.push(s);
            }
        }
        self.springs = keep_sp;

        // Drop triangles that lost any structural edge. A triangle's structural
        // edges are its three sides; an edge is "structural" if it survives as a
        // distance constraint (the XPBD path used by cloth) before this tear.
        // (MassSpring cloth is built from springs, which we also dropped above, so
        // we check both: an edge that is neither a surviving distance-constraint
        // nor a surviving spring has snapped.)
        let has_edge = |a: u32, b: u32| -> bool {
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            self.distance_constraints.iter().any(|c| {
                let (ca, cb) = if c.a as u32 <= c.b as u32 {
                    (c.a as u32, c.b as u32)
                } else {
                    (c.b as u32, c.a as u32)
                };
                (ca, cb) == (lo, hi)
            }) || self.springs.iter().any(|s| {
                let (sa, sb) = if s.a as u32 <= s.b as u32 {
                    (s.a as u32, s.b as u32)
                } else {
                    (s.b as u32, s.a as u32)
                };
                (sa, sb) == (lo, hi)
            })
        };
        self.triangles
            .retain(|t| has_edge(t[0], t[1]) && has_edge(t[1], t[2]) && has_edge(t[2], t[0]));
        let _ = broken_dc; // retained for clarity; broken edges already excluded.
    }

    /// Removes the particle at `index`, keeping the topology consistent.
    ///
    /// Springs, distance constraints, and tetrahedra that *reference* the removed
    /// particle are dropped; every remaining index `> index` is decremented by one
    /// so it still points at the same particle. `tetra_rest_volumes` is filtered in
    /// lockstep with `tetrahedra`. Returns `false` (and does nothing) if `index` is
    /// out of bounds.
    ///
    /// This is the per-particle counterpart of [`SoftBodySet::remove`]: deleting a
    /// block/voxel in a Minecraft chunk maps to removing the corresponding particle
    /// plus its incident springs/edges, after which the body keeps simulating under
    /// the new (smaller) topology.
    pub fn remove_particle(&mut self, index: usize) -> bool {
        if index >= self.particles.len() {
            return false;
        }
        self.particles.remove(index);

        // Springs: drop any touching `index`, shift the rest.
        self.springs.retain_mut(|s| {
            if s.a == index || s.b == index {
                return false;
            }
            if s.a > index {
                s.a -= 1;
            }
            if s.b > index {
                s.b -= 1;
            }
            true
        });

        // Distance constraints: same treatment.
        self.distance_constraints.retain_mut(|c| {
            if c.a == index || c.b == index {
                return false;
            }
            if c.a > index {
                c.a -= 1;
            }
            if c.b > index {
                c.b -= 1;
            }
            true
        });

        // Tetrahedra: drop any containing `index`; otherwise shift each index down.
        let keep: Vec<bool> = self
            .tetrahedra
            .iter()
            .map(|t| !t.contains(&(index as u32)))
            .collect();
        let remapped: Vec<[u32; 4]> = self
            .tetrahedra
            .iter()
            .enumerate()
            .filter(|(i, _)| keep[*i])
            .map(|(_, t)| {
                let mut t = *t;
                for x in t.iter_mut() {
                    if *x as usize > index {
                        *x -= 1;
                    }
                }
                t
            })
            .collect();
        let kept_volumes: Vec<Real> = self
            .tetra_rest_volumes
            .iter()
            .enumerate()
            .filter(|(i, _)| keep[*i])
            .map(|(_, v)| *v)
            .collect();
        self.tetrahedra = remapped;
        self.tetra_rest_volumes = kept_volumes;

        // Triangles: drop any containing `index`; otherwise shift each index down.
        let keep_tri: Vec<bool> = self
            .triangles
            .iter()
            .map(|t| !t.contains(&(index as u32)))
            .collect();
        let remapped_tri: Vec<[u32; 3]> = self
            .triangles
            .iter()
            .enumerate()
            .filter(|(i, _)| keep_tri[*i])
            .map(|(_, t)| {
                let mut t = *t;
                for x in t.iter_mut() {
                    if *x as usize > index {
                        *x -= 1;
                    }
                }
                t
            })
            .collect();
        self.triangles = remapped_tri;

        true
    }

    /// Total (sum of absolute) signed volume of all tetrahedra — a finite,
    /// deformation-sensitive scalar useful for regression tests.
    pub fn total_volume(&self) -> Real {
        self.tetra_rest_volumes
            .iter()
            .zip(self.tetrahedra.iter())
            .map(|(rest_vol, tet)| {
                let [a, b, c, d] = *tet;
                let (pa, pb, pc, pd) = (
                    self.particles[a as usize].pos,
                    self.particles[b as usize].pos,
                    self.particles[c as usize].pos,
                    self.particles[d as usize].pos,
                );
                signed_tetra_volume(pa, pb, pc, pd).abs() / rest_vol.abs().max(1e-12)
            })
            .fold(0.0, |acc, r| acc + r)
    }

    /// Phase 3 XPBD substep (Matthias Müller "Small Steps" XPBD):
    ///
    /// 1. **Predict**: for each free particle, `v += dt·g`, `x_prev = x`, `x += dt·v`.
    /// 2. **Project** `iterations` times (Gauss-Seidel, fixed constraint order for
    ///    determinism): each distance constraint then each tetra volume constraint
    ///    is solved, accumulating per-constraint Lagrange multipliers `λ`.
    /// 3. **Update velocities**: `v = (x − x_prev) / dt`.
    ///
    /// Bound particles (`bound_body.is_some()`) are treated as infinite mass
    /// (effective inverse mass 0) so the soft body can be anchored to rigid bodies
    /// without the XPBD solve fighting `force_containers` (Phase 2).
    ///
    /// **Determinism**: constraints are traversed in vector order with no
    /// concurrency, so two runs from identical state are bit-identical. (Compliance
    /// `α̃ = α / dt²` makes stiff edges stable; `α = 0` gives a hard constraint.)
    pub fn step_xpbd(&mut self, dt: Real) {
        if self.sleeping {
            return;
        }
        let iterations = match self.solver {
            SoftSolver::Xpbd { iterations, .. } => iterations,
            SoftSolver::MassSpring => return, // guarded by step(), defensive
        };
        if iterations == 0 {
            return;
        }
        let n = self.particles.len();
        // Effective inverse mass: 0 for pinned **and** bound (rigid-anchored) particles.
        let mut w = Vec::with_capacity(n);
        for p in &self.particles {
            w.push(if p.bound_body.is_some() {
                0.0
            } else {
                p.inv_mass
            });
        }
        // Previous positions (for velocity recovery).
        let mut prev = Vec::with_capacity(n);
        for p in &self.particles {
            prev.push(p.pos);
        }

        // 1. Predict.
        for (i, p) in self.particles.iter_mut().enumerate() {
            if w[i] == 0.0 {
                continue;
            }
            // Phase 7: wind / air-resistance is a pure external acceleration,
            // applied alongside gravity in the predict step (no new mechanics).
            let mut a = self.gravity;
            if let Some(wind) = self.wind {
                a += wind.accel;
                a -= wind.drag * p.vel;
            }
            p.vel += dir_scaled(a, dt);
            p.pos += dir_scaled(p.vel, dt);
        }

        // 2. Project (fixed order → deterministic).
        let alpha = self.xpbd_alpha(dt);
        // Extract constraint parameters into local buffers so the projection loop
        // can mutate `self.particles` through `&mut self` without holding an
        // immutable borrow of `self.distance_constraints` / `self.tetrahedra`
        // (avoids the borrow checker's simultaneous mutable+immutable error).
        // Order is preserved → determinism is unchanged.
        let ndc = self.distance_constraints.len();
        let mut d_a = Vec::with_capacity(ndc);
        let mut d_b = Vec::with_capacity(ndc);
        let mut d_rest = Vec::with_capacity(ndc);
        for c in &self.distance_constraints {
            d_a.push(c.a);
            d_b.push(c.b);
            d_rest.push(c.rest);
        }
        let mut d_lambda = Vec::with_capacity(ndc);
        #[allow(clippy::same_item_push)] // Lagrange accumulator, filled with zeros
        for _ in 0..ndc {
            d_lambda.push(0.0);
        }
        let ntet = self.tetrahedra.len();
        let mut t_idx = Vec::with_capacity(ntet);
        let mut t_rest = Vec::with_capacity(ntet);
        for (i, tet) in self.tetrahedra.iter().enumerate() {
            t_idx.push(*tet);
            t_rest.push(self.tetra_rest_volumes[i]);
        }
        let mut v_lambda = Vec::with_capacity(ntet);
        #[allow(clippy::same_item_push)] // Lagrange accumulator, filled with zeros
        for _ in 0..ntet {
            v_lambda.push(0.0);
        }
        for _ in 0..iterations {
            for ci in 0..ndc {
                solve_distance_constraint(
                    &mut self.particles,
                    d_a[ci],
                    d_b[ci],
                    d_rest[ci],
                    alpha,
                    &w,
                    &mut d_lambda[ci],
                );
            }
            for ti in 0..ntet {
                solve_volume_constraint(
                    &mut self.particles,
                    &t_idx[ti],
                    t_rest[ti],
                    alpha,
                    &w,
                    &mut v_lambda[ti],
                );
            }
        }

        // 3. Recover velocities.
        for (i, p) in self.particles.iter_mut().enumerate() {
            if w[i] == 0.0 {
                continue;
            }
            p.vel = (p.pos - prev[i]) / dt;
        }
    }

    /// XPBD `α̃ = α / dt²` from the current solver compliance.
    fn xpbd_alpha(&self, dt: Real) -> Real {
        let compliance = match self.solver {
            SoftSolver::Xpbd { compliance, .. } => compliance,
            SoftSolver::MassSpring => 0.0,
        };
        compliance / (dt * dt)
    }
}

/// `v * s` for a `Vector` and scalar `s` (glam supports `Vec3 * f64`).
#[inline]
fn dir_scaled(v: Vector, s: Real) -> Vector {
    v * s
}

// ── Phase 3: XPBD constraint projections ───────────────────────────────────
//
// These are free functions (not methods) so the solver can mutate particle
// positions directly without borrow fights. All arithmetic is plain `f64`
// four-operations on glam `Vector`s — bit-identical under IEEE-754, which is
// what makes XPBD reproducible for `enhanced-determinism` (no `linalg` matrix
// solve is needed for position-based projection; `linalg` is reserved for the
// optional implicit-FEM comparison path in a later sub-phase).

/// Signed volume of the tetrahedron `(p0, p1, p2, p3)`:
/// `V = ((p1−p0) × (p2−p0)) · (p3−p0) / 6`.
#[inline]
fn signed_tetra_volume(p0: Vector, p1: Vector, p2: Vector, p3: Vector) -> Real {
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let e3 = p3 - p0;
    e1.cross(e2).dot(e3) / 6.0
}

/// Solve one XPBD distance constraint, updating `particles` in place and
/// accumulating the Lagrange multiplier into `*lambda`.
#[inline]
fn solve_distance_constraint(
    particles: &mut [SoftParticle],
    a: usize,
    b: usize,
    rest: Real,
    alpha: Real,
    w: &[Real],
    lambda: &mut Real,
) {
    let pa = particles[a].pos;
    let pb = particles[b].pos;
    let delta = pa - pb; // vector from b → a (XPBD standard: d = p_a − p_b)
    let len = delta.length();
    if len == 0.0 {
        return;
    }
    let n = delta / len; // points from b toward a
    let c_val = len - rest;
    let wa = w[a];
    let wb = w[b];
    let w_sum = wa + wb;
    if w_sum == 0.0 {
        return;
    }
    let d_lambda = (-c_val - alpha * *lambda) / (w_sum + alpha);
    *lambda += d_lambda;
    // Standard XPBD distance projection: p_a += w_a·Δλ·n, p_b −= w_b·Δλ·n.
    // With C = len − rest > 0 (too long), Δλ < 0, so a moves toward b and b toward a.
    if wa != 0.0 {
        particles[a].pos += dir_scaled(n, wa * d_lambda);
    }
    if wb != 0.0 {
        particles[b].pos -= dir_scaled(n, wb * d_lambda);
    }
}

/// Solve one XPBD tetrahedral volume constraint, updating `particles` in
/// place and accumulating the Lagrange multiplier into `*lambda`.
///
/// Constraint: `C = V − V0` where `V` is the current signed volume. Gradients
/// (Müller et al.):
/// `∇_0 = −(e2×e3)/6`, `∇_1 = (e3×e1)/6`, `∇_2 = (e1×e2)/6`, `∇_3 = −(e1×e3)/6`
/// (with `e_i = p_i − p_0`). The correction `Δp_i = w_i · Δλ · ∇_i`.
#[inline]
fn solve_volume_constraint(
    particles: &mut [SoftParticle],
    tet: &[u32; 4],
    rest_vol: Real,
    alpha: Real,
    w: &[Real],
    lambda: &mut Real,
) {
    let [a, b, c, d] = *tet;
    let (ia, ib, ic, id) = (a as usize, b as usize, c as usize, d as usize);
    let p0 = particles[ia].pos;
    let p1 = particles[ib].pos;
    let p2 = particles[ic].pos;
    let p3 = particles[id].pos;
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let e3 = p3 - p0;

    let vol = e1.cross(e2).dot(e3) / 6.0;
    let c_val = vol - rest_vol;

    let g0 = e2.cross(e3) / -6.0;
    let g1 = e3.cross(e1) / 6.0;
    let g2 = e1.cross(e2) / 6.0;
    let g3 = e1.cross(e3) / -6.0;

    let wa = w[ia];
    let wb = w[ib];
    let wc = w[ic];
    let wd = w[id];
    // Σ w_i |∇_i|²  (all four gradients; w=0 particles contribute 0).
    let mut denom = alpha;
    denom += wa * g0.dot(g0);
    denom += wb * g1.dot(g1);
    denom += wc * g2.dot(g2);
    denom += wd * g3.dot(g3);
    if denom == 0.0 {
        return;
    }
    let d_lambda = (-c_val - alpha * *lambda) / denom;
    *lambda += d_lambda;

    if wa != 0.0 {
        particles[ia].pos += dir_scaled(g0, wa * d_lambda);
    }
    if wb != 0.0 {
        particles[ib].pos += dir_scaled(g1, wb * d_lambda);
    }
    if wc != 0.0 {
        particles[ic].pos += dir_scaled(g2, wc * d_lambda);
    }
    if wd != 0.0 {
        particles[id].pos += dir_scaled(g3, wd * d_lambda);
    }
}

/// A container owning all soft bodies in a simulation. Phase 0 keeps this as a
/// plain `Vec` store; later phases may back it with the arena used by the
/// rigid-body / joint sets.
#[derive(Clone, Debug, Default)]
pub struct SoftBodySet {
    /// Slot storage. A slot becomes `None` once its body is removed via
    /// [`SoftBodySet::remove`], which keeps `SoftBodyId`s stable: ids are slot
    /// indices, so removal is id-preserving and never reshuffles live bodies
    /// (important for the FFI layer that hands `SoftBodyId`s to callers).
    bodies: Vec<Option<SoftBody>>,
}

impl SoftBodySet {
    /// Creates an empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts a soft body and returns its id.
    pub fn insert(&mut self, body: SoftBody) -> SoftBodyId {
        let id = SoftBodyId(self.bodies.len() as u32);
        self.bodies.push(Some(body));
        id
    }

    /// Number of slots (including tombstoned/removed ones).
    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    /// Number of live (not removed) soft bodies.
    pub fn count(&self) -> usize {
        self.bodies.iter().filter(|b| b.is_some()).count()
    }

    /// Whether the set holds no live bodies.
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Removes the soft body with the given id, freeing its storage. The id
    /// stays reserved (the slot becomes a tombstone), so every other live
    /// `SoftBodyId` remains valid — callers may keep holding ids across removals.
    /// Returns `true` if a live body was removed.
    pub fn remove(&mut self, id: SoftBodyId) -> bool {
        let slot = match self.bodies.get_mut(id.0 as usize) {
            Some(slot) => slot,
            None => return false,
        };
        match slot.take() {
            Some(_) => true,
            None => false,
        }
    }

    /// Immutable access by id. Returns `None` for an unknown or removed id.
    #[allow(dead_code)] // consumed by later integration phases (World/FFI).
    pub fn get(&self, id: SoftBodyId) -> Option<&SoftBody> {
        self.bodies.get(id.0 as usize).and_then(|b| b.as_ref())
    }

    /// Mutable access by id. Returns `None` for an unknown or removed id.
    pub fn get_mut(&mut self, id: SoftBodyId) -> Option<&mut SoftBody> {
        self.bodies.get_mut(id.0 as usize).and_then(|b| b.as_mut())
    }

    /// Advances every live soft body by `dt` (sleeping bodies are skipped).
    pub fn step(&mut self, dt: Real) {
        for body in self.bodies.iter_mut().flatten() {
            body.step(dt);
        }
    }

    /// Marks a soft body as sleeping (no further integration until woken).
    pub fn sleep(&mut self, id: SoftBodyId) -> bool {
        match self.bodies.get_mut(id.0 as usize).and_then(|b| b.as_mut()) {
            Some(b) => {
                b.sleeping = true;
                true
            }
            None => false,
        }
    }

    /// Wakes a sleeping soft body.
    pub fn wake(&mut self, id: SoftBodyId) -> bool {
        match self.bodies.get_mut(id.0 as usize).and_then(|b| b.as_mut()) {
            Some(b) => {
                b.sleeping = false;
                true
            }
            None => false,
        }
    }

    /// Whether the soft body is currently sleeping.
    pub fn is_sleeping(&self, id: SoftBodyId) -> bool {
        self.bodies
            .get(id.0 as usize)
            .and_then(|b| b.as_ref())
            .map(|b| b.sleeping)
            .unwrap_or(false)
    }

    /// Phase 8: for every live soft body, snap each bound particle to its rigid
    /// body's current world transform (`pos = body_local → world`, `vel =
    /// body.velocity_at_point(world)`). Bound particles are infinite-mass in the
    /// XPBD solve and skipped by local integration, so this is what makes a
    /// particle *rigidly follow* the body it is anchored to (flags, tethers,
    /// cloth pinned to a moving object). Call once per step, before
    /// [`Self::step`] so the followers are already in place when constraints
    /// project. Skips sleeping bodies.
    pub fn follow_rigid_bodies(&mut self, bodies: &RigidBodySet) {
        for body in self.bodies.iter_mut().flatten() {
            if body.sleeping {
                continue;
            }
            for p in body.particles.iter_mut() {
                let Some(h) = p.bound_body else {
                    continue;
                };
                let Some(rb) = bodies.get(h) else {
                    continue;
                };
                let world = rb.position().transform_point(p.bound_local);
                p.pos = world;
                p.vel = rb.velocity_at_point(world);
            }
        }
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
        for body in self.bodies.iter().flatten() {
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

    // ── Phase 3: XPBD ─────────────────────────────────────────────────────

    #[test]
    fn xpbd_distance_constraint_restores_length() {
        // Two free particles at distance 1 (rest=1), then b is yanked out to
        // distance 4. XPBD projection should pull the gap back toward rest.
        let mut body = SoftBody::new(Vector::ZERO);
        let a = body.add_particle(Vector::new(0.0, 0.0, 0.0));
        let b = body.add_particle(Vector::new(1.0, 0.0, 0.0));
        body.configure_xpbd(20, 0.0); // rigid (α=0)
        body.add_distance_constraint(a, b, 0.0); // rest captured as 1.0

        // Perturb b to distance 4 (current > rest).
        body.particles[b].pos = Vector::new(4.0, 0.0, 0.0);
        let d0 = (body.particles[b].pos - body.particles[a].pos).length();
        body.step_xpbd(0.01);
        let d1 = (body.particles[b].pos - body.particles[a].pos).length();
        // Distance should shrink from 4 toward rest (1.0).
        assert!(d1 < d0, "XPBD distance should shrink gap: {d0} -> {d1}");
        assert!(d1 > 0.5 && d1 < 4.0, "XPBD gap in sane band: {d1}");
        assert!(body.particles[a].pos.is_finite());
        assert!(body.particles[b].pos.is_finite());
    }

    #[test]
    fn xpbd_volume_constraint_preserves_tetrahedron() {
        // Regular tetrahedron; perturb one vertex, then XPBD volume constraint
        // should keep the (relative) volume finite and pull it back toward rest.
        let mut body = SoftBody::new(Vector::ZERO);
        let p0 = body.add_particle(Vector::new(0.0, 0.0, 0.0));
        let p1 = body.add_particle(Vector::new(1.0, 0.0, 0.0));
        let p2 = body.add_particle(Vector::new(0.0, 1.0, 0.0));
        let p3 = body.add_particle(Vector::new(0.0, 0.0, 1.0));
        body.configure_xpbd(20, 0.0);
        body.add_tetrahedron([p0 as u32, p1 as u32, p2 as u32, p3 as u32]);
        let rest = body.total_volume();
        assert!(rest.is_finite() && rest > 0.0, "rest volume sane: {rest}");

        // Perturb p3 outward; volume should grow, then projection pulls it back.
        body.particles[p3].pos = Vector::new(0.0, 0.0, 5.0);
        let perturbed = body.total_volume();
        assert!(
            perturbed > rest,
            "perturbation grew volume: {rest} -> {perturbed}"
        );

        body.step_xpbd(0.01);
        let recovered = body.total_volume();
        // Should be pulled back toward rest (not explode, not collapse to 0).
        assert!(recovered.is_finite());
        assert!(
            recovered > 0.1 * rest && recovered < perturbed,
            "volume recovered: {rest} -> {perturbed} -> {recovered}"
        );
    }

    #[test]
    fn xpbd_is_deterministic_bit_identical() {
        // Two identical XPBD bodies from the same initial state must produce
        // bit-identical results (fixed constraint order, IEEE-754 float ops).
        let build = || {
            let mut body = SoftBody::new(Vector::new(0.0, -9.81, 0.0));
            let p0 = body.add_particle(Vector::new(0.0, 0.0, 0.0));
            let p1 = body.add_particle(Vector::new(1.0, 0.0, 0.0));
            let p2 = body.add_particle(Vector::new(0.0, 1.0, 0.0));
            let p3 = body.add_particle(Vector::new(0.0, 0.0, 1.0));
            body.configure_xpbd(15, 1e-6);
            body.add_distance_constraint(p0, p1, 1e-6);
            body.add_distance_constraint(p1, p2, 1e-6);
            body.add_distance_constraint(p2, p3, 1e-6);
            body.add_tetrahedron([p0 as u32, p1 as u32, p2 as u32, p3 as u32]);
            body
        };
        let mut a = build();
        let mut b = build();
        for _ in 0..30 {
            a.step_xpbd(0.01);
            b.step_xpbd(0.01);
        }
        for i in 0..a.particles.len() {
            assert_eq!(
                a.particles[i].pos.x.to_bits(),
                b.particles[i].pos.x.to_bits(),
                "x bit-identical p{i}"
            );
            assert_eq!(
                a.particles[i].pos.y.to_bits(),
                b.particles[i].pos.y.to_bits(),
                "y bit-identical p{i}"
            );
            assert_eq!(
                a.particles[i].pos.z.to_bits(),
                b.particles[i].pos.z.to_bits(),
                "z bit-identical p{i}"
            );
        }
    }

    #[test]
    fn xpbd_step_dispatcher_routes_to_xpbd() {
        // `SoftBody::step` must dispatch to XPBD when solver is configured.
        let mut body = SoftBody::new(Vector::ZERO);
        let a = body.add_particle(Vector::new(0.0, 0.0, 0.0));
        let b = body.add_particle(Vector::new(1.0, 0.0, 0.0));
        body.configure_xpbd(20, 0.0); // rest captured as 1.0
        body.add_distance_constraint(a, b, 0.0);
        body.particles[b].pos = Vector::new(4.0, 0.0, 0.0); // yank to 4
        let d0 = (body.particles[b].pos - body.particles[a].pos).length();
        body.step(0.01); // dispatches to step_xpbd
        let d1 = (body.particles[b].pos - body.particles[a].pos).length();
        assert!(d1 < d0, "dispatched XPBD should shrink gap: {d0} -> {d1}");
    }
}
