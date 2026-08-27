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
use std::collections::{HashMap, HashSet};
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
    /// XPBD stretch compliance `α_s` (0 = rigid, > 0 = soft), applied when the edge
    /// is **longer** than `rest` (tension). Stored per-constraint so different edges
    /// can have different stiffness. Phase 19: this is the *stretch* compliance; the
    /// *compression* compliance lives in [`Self::compression`], enabling anisotropic
    /// behaviour (e.g. cloth resists stretch but folds easily under compression).
    pub compliance: Real,
    /// Phase 19 — XPBD compression compliance `α_c` (0 = rigid, > 0 = soft), applied
    /// when the edge is **shorter** than `rest` (compression). When equal to
    /// [`Self::compliance`] the edge is isotropic. Initialized to the stretch
    /// compliance by every constructor (`add_distance_constraint`, `add_triangle`,
    /// `add_bending_constraint`) so existing bodies stay isotropic unless the caller
    /// opts into anisotropy via `set_distance_constraint_compression`.
    pub compression: Real,
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
    /// Phase 11: uniform internal pressure (`P`, force/area). When `Some`, every
    /// free particle of a closed triangular mesh gets an outward push along the
    /// surface normal, `F = P · area` per incident triangle (balloon / gas-law
    /// model). A pure external force mirroring [`Self::wind`]; applied in both the
    /// MassSpring (`compute_forces`) and XPBD (`step_xpbd` predict) paths. `None`
    /// (default) = no pressure. Closed manifold (`self.triangles`) needed for a
    /// real balloon; an open sheet just bulges along single-sided normals.
    pub pressure: Option<Real>,
    /// Phase 18: global internal (structural) damping. Each step, every free
    /// particle's velocity is scaled by `1 − damping` (after the solver recovers
    /// velocities), giving a body-wide "jelly / slime" energy loss that is
    /// orthogonal to the per-spring axial `damping` (Phase 0) and to the
    /// distance-solver `compliance` (Phase 13). `0` (default) = no damping (energy
    /// conserving for the internal modes); larger values settle oscillation / jitter
    /// faster. Clamped to `[0, 1)` — `1` would fully freeze motion, so it is rejected.
    /// Applied in both the MassSpring (`integrate`) and XPBD (`step_xpbd` velocity
    /// recovery) paths. No new solver mechanics, no SoA interaction.
    pub damping: Real,
    /// Phase 24: XPBD/MassSpring substeps per [`SoftBody::step`] call. Splits the
    /// frame `dt` into `substeps` equal slices and runs the active solver once per
    /// slice (so constraint projection happens at a finer time resolution). `1`
    /// reproduces the historic single-substep behaviour. Larger values make stiff
    /// materials / high compliance converge faster and stay stable; they cost
    /// `substeps`× the per-step work. Mirrors `World::integration_parameters`
    /// substepping for the rigid side — same idea, local to each soft body.
    pub substeps: u32,
    /// Phase 9: tearing threshold. When `Some(ε)`, any structural edge (XPBD
    /// distance constraint or MassSpring spring) whose strain `(|len| − rest)/rest`
    /// exceeds `ε` is removed at the start of each [`SoftBody::step`]. Triangular
    /// faces that lose any structural edge are dropped too, so a torn cloth stops
    /// rendering the broken face. `None` (default) = no tearing. Pure topology
    /// edit — no new solver mechanics, no SoA interaction.
    pub tear_strain: Option<Real>,
    /// Phase 10: plasticity (permanent deformation, like putty / memory foam).
    /// When `Some(params)`, any structural edge whose elastic strain magnitude
    /// exceeds `params.yield_strain` has its rest length permanently shifted
    /// toward the current length by `params.creep` (clamped to `[0,1]`) each step,
    /// so the deformation "freezes in" instead of springing back. `None` (default)
    /// = perfectly elastic (Hookean). Pure rest-length edit — no new solver
    /// mechanics, no SoA interaction.
    pub plasticity: Option<PlasticityParams>,
    /// Phase 12: self-collision. When `Some(params)`, the body's free particles
    /// repel each other when their centres come within `2·params.radius` (each
    /// particle behaves as a sphere of that radius). Broad-phase uses a uniform
    /// spatial hash; detected pairs are solved as stiff XPBD distance constraints
    /// (rest = `2·radius`, compliance = `params.stiffness`) every solver iteration,
    /// in both the MassSpring and XPBD paths. Direct structural neighbours (linked
    /// by a [`SoftBody::distance_constraints`] edge) are excluded so existing
    /// springs/cloth edges are not treated as collisions. `None` (default) = off.
    /// Pure positional projection — no new solver mechanics, no SoA interaction.
    pub self_collision: Option<SelfCollisionParams>,
    /// Phase 14: soft-soft (cross-body) collision. When `Some(params)`, this body
    /// collides with *other* soft bodies that also have `cross_collision` set: their
    /// free particles repel when centres come within `2·min(radius_a, radius_b)`.
    /// Reuses the same spatial-hash broad-phase + XPBD push-apart as self-collision,
    /// but the world-level pass runs over pairs of bodies. `None` (default) = off.
    /// Pure positional projection — no new solver mechanics, no SoA interaction.
    pub cross_collision: Option<SelfCollisionParams>,
    /// Phase 16: dedicated volume-conservation compliance for the tetrahedral
    /// volume constraints. When `Some(c)`, every tetra volume constraint in
    /// `step_xpbd` is solved with `α̃ = c / dt²` — *independent* of the distance
    /// solver's compliance. This makes it possible to have soft edges but a hard
    /// (incompressible) blob, or to keep volume conserved even when the distance
    /// solver is soft. It is orthogonal to Phase 11 pressure (an outward force):
    /// pressure inflates, this constraint holds the total volume. `None` (default)
    /// = fall back to the global solver compliance for tetra volume (existing
    /// behaviour). Pure positional constraint — no new solver mechanics.
    pub volume_conservation: Option<Real>,
    /// Phase 17: cohesion (adhesion / breakable glue) between this body and *other*
    /// bodies. When `Some(p)`, free particles of this body within `p.radius` of a
    /// free particle of another cohesion-enabled body attract toward contact, bonding
    /// the bodies (the dual of Phase 9 tearing). Bonds break when pulled apart beyond
    /// `p.break_distance`. Solved at world level by `solve_cohesion`. `None` (default)
    /// = off.
    pub cohesion: Option<CohesionParams>,
}
/// Phase 10: plasticity parameters (see [`SoftBody::plasticity`]).
#[derive(Clone, Copy, Debug)]
pub struct PlasticityParams {
    /// Yield strain: elastic deformation below this magnitude is fully recovered;
    /// above it, the excess becomes permanent. Must be ≥ 0.
    pub yield_strain: Real,
    /// Creep rate in `[0,1]`: fraction of the over-yield strain that is transferred
    /// from elastic to plastic (rest-length) each step. `1` = instantly frozen at
    /// the yield surface; `0` = no plasticity (elastic).
    pub creep: Real,
}

/// Phase 12: self-collision parameters (see [`SoftBody::self_collision`]).
#[derive(Clone, Copy, Debug)]
pub struct SelfCollisionParams {
    /// Particle collision radius. Two *free* particles whose centres come within
    /// `2·radius` are pushed apart (each treated as a sphere of this radius). Should
    /// match `SoftBody::particle_radius` used by the proxy-collider path, but is
    /// independent here so self-collision works without rigin-body coupling. Must be > 0.
    pub radius: Real,
    /// XPBD compliance of the repulsion constraint. `0` = perfectly hard (rigid
    /// non-penetration); larger values allow softer, springier contact. Must be ≥ 0.
    pub stiffness: Real,
    /// Phase 20: contact friction coefficient for the tangential relative slip at a
    /// soft-soft contact (self-collision and, via the shared struct, cross-collision).
    /// `None` = frictionless (default). When set, the tangential relative velocity of a
    /// contacting pair is damped by `μ` each step (Coulomb-style, bounded to `[0,1]`:
    /// `μ = 0` no friction, `μ = 1` fully kills tangential slip). Must be `0 ≤ μ ≤ 1`.
    pub friction: Option<Real>,
}

/// Phase 17: cohesion (adhesion / breakable glue) parameters for inter-body
/// (soft-soft) contact — the dual of Phase 9 tearing. Two *free* particles from
/// *different* bodies whose centres come within `radius` attract toward contact
/// (rest distance `radius`), bonding the bodies together like glue. The bond is
/// *breakable*: if the pair separation ever exceeds `break_distance` (which is
/// usually `> radius`, giving a hysteresis so bonded pairs need to be pulled apart
/// to break), the attraction is released for that pair for the rest of the step —
/// i.e. the glue tears. Each step is stateless: bonds are re-evaluated from the
/// current geometry, so it composes naturally with Phase 14 cross-collision.
#[derive(Clone, Copy, Debug)]
pub struct CohesionParams {
    /// Capture radius: a free particle from another body within this distance is
    /// attracted and bonded. Should match `SoftBody::particle_radius` conceptually.
    /// Must be > 0.
    pub radius: Real,
    /// XPBD compliance of the attraction constraint. `0` = hard glue (bonded pairs
    /// snap to exactly `radius` apart); larger values give springier, stretchier
    /// glue. Must be ≥ 0.
    pub stiffness: Real,
    /// Break distance: once a bonded pair is pulled apart beyond this separation the
    /// attraction releases (the glue tears). Must be `> radius`. `inf` disables
    /// breaking (permanent glue).
    pub break_distance: Real,
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
            plasticity: None,
            pressure: None,
            damping: 0.0,
            self_collision: None,
            cross_collision: None,
            volume_conservation: None,
            cohesion: None,
            substeps: 1,
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

    /// Phase 11: enables a uniform internal pressure `P` (force/area). When `P > 0`,
    /// every free particle of a closed triangular mesh is pushed outward along the
    /// surface normal with force `F = P · area` per incident triangle (the balloon /
    /// gas-law model). Pass `P <= 0` (or [`Self::clear_pressure`]) to disable.
    pub fn set_pressure(&mut self, pressure: Real) {
        if pressure > 0.0 {
            self.pressure = Some(pressure);
        } else {
            self.pressure = None;
        }
    }

    /// Phase 11: disables internal pressure (`None`).
    pub fn clear_pressure(&mut self) {
        self.pressure = None;
    }

    /// Phase 11: per-particle outward pressure forces, `F_i = Σ_t P · area(t) · n̂(t)`
    /// over triangles incident to `i`. The normal is the *centroid-oriented* outward
    /// direction: each triangle contributes equally to its three vertices using the
    /// face normal that points away from the mesh centroid. This keeps a closed mesh
    /// inflating symmetrically (no net single-sided bias) and lets an open sheet
    /// bulge along its normals. Pure topology read — no solver state touched.
    fn pressure_forces(&self) -> Vec<Vector> {
        let p = match self.pressure {
            Some(p) => p,
            None => {
                return {
                    let mut v = Vec::with_capacity(self.particles.len());
                    for _ in 0..self.particles.len() {
                        v.push(Vector::ZERO);
                    }
                    v
                };
            }
        };
        // Mesh centroid for outward orientation.
        let centroid = if self.particles.is_empty() {
            Vector::ZERO
        } else {
            let mut c = Vector::ZERO;
            for pt in &self.particles {
                c += pt.pos;
            }
            c / (self.particles.len() as Real)
        };
        let mut forces = Vec::with_capacity(self.particles.len());
        for _ in 0..self.particles.len() {
            forces.push(Vector::ZERO);
        }
        for tri in &self.triangles {
            let (ia, ib, ic) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            let (pa, pb, pc) = match (
                self.particles.get(ia),
                self.particles.get(ib),
                self.particles.get(ic),
            ) {
                (Some(a), Some(b), Some(c)) => (a.pos, b.pos, c.pos),
                _ => continue,
            };
            // Face normal (not normalized) = (b−a) × (c−a); magnitude = 2·area.
            let n = (pb - pa).cross(pc - pa);
            let area = n.length() * 0.5;
            if area <= 0.0 {
                continue;
            }
            // Orient outward relative to centroid; n̂ is the unit face normal.
            let n_hat = n.normalize();
            let tri_center = (pa + pb + pc) / 3.0;
            let outward = if (tri_center - centroid).dot(n_hat) >= 0.0 {
                n_hat
            } else {
                -n_hat
            };
            // Force magnitude per triangle: P · area, split equally to 3 vertices.
            let f = outward * (p * area / 3.0);
            if let Some(fa) = forces.get_mut(ia) {
                *fa += f;
            }
            if let Some(fb) = forces.get_mut(ib) {
                *fb += f;
            }
            if let Some(fc) = forces.get_mut(ic) {
                *fc += f;
            }
        }
        forces
    }

    /// Phase 12: enables self-collision with the given `radius` (particle sphere
    /// radius) and `stiffness` (XPBD compliance of the repulsion constraint, `0`
    /// = hard). Pairs of free particles closer than `2·radius` are pushed apart
    /// every solver iteration. Rejects non-positive `radius` or negative
    /// `stiffness` (returns `false` without enabling).
    pub fn set_self_collision(&mut self, radius: Real, stiffness: Real) -> bool {
        if !(radius > 0.0) || stiffness < 0.0 {
            return false;
        }
        self.self_collision = Some(SelfCollisionParams {
            radius,
            stiffness,
            friction: None,
        });
        true
    }

    /// Phase 20: sets the contact friction coefficient `μ` for self-collision.
    /// Requires `self_collision` to be enabled first. Rejects non-finite or out-of-range
    /// `μ` (`0 ≤ μ ≤ 1`). `clear_self_collision` resets it to `None` (frictionless).
    pub fn set_self_collision_friction(&mut self, mu: Real) -> bool {
        if !mu.is_finite() || mu < 0.0 || mu > 1.0 {
            return false;
        }
        let Some(p) = self.self_collision.as_mut() else {
            return false;
        };
        p.friction = Some(mu);
        true
    }

    /// Phase 12: disables self-collision (`None`).
    pub fn clear_self_collision(&mut self) {
        self.self_collision = None;
    }

    /// Phase 14: enables soft-soft (cross-body) collision with the given `radius`
    /// (particle sphere radius) and `stiffness` (XPBD compliance of the repulsion
    /// constraint, `0` = hard). Two bodies only collide if *both* have
    /// `cross_collision` set; the effective repulsion distance is
    /// `2·min(radius_a, radius_b)`. Rejects non-positive `radius` or negative
    /// `stiffness`.
    pub fn set_cross_collision(&mut self, radius: Real, stiffness: Real) -> bool {
        if !(radius > 0.0) || stiffness < 0.0 {
            return false;
        }
        self.cross_collision = Some(SelfCollisionParams {
            radius,
            stiffness,
            friction: None,
        });
        true
    }

    /// Phase 20: sets the contact friction coefficient `μ` for cross-collision.
    /// Requires `cross_collision` to be enabled first. Rejects non-finite or out-of-range
    /// `μ` (`0 ≤ μ ≤ 1`). `clear_cross_collision` resets it to `None` (frictionless).
    pub fn set_cross_collision_friction(&mut self, mu: Real) -> bool {
        if !mu.is_finite() || mu < 0.0 || mu > 1.0 {
            return false;
        }
        let Some(p) = self.cross_collision.as_mut() else {
            return false;
        };
        p.friction = Some(mu);
        true
    }

    /// Phase 14: disables soft-soft (cross-body) collision (`None`).
    pub fn clear_cross_collision(&mut self) {
        self.cross_collision = None;
    }

    /// Phase 16: enables the dedicated volume-conservation constraint with the given
    /// `compliance` (`c`). Each tetra volume constraint in `step_xpbd` is then solved
    /// with `α̃ = c / dt²`, independent of the distance solver's compliance. `c == 0`
    /// gives a hard (incompressible) blob. Returns `false` (and does nothing) on a
    /// non-finite or negative `compliance`.
    pub fn set_volume_conservation(&mut self, compliance: Real) -> bool {
        if !compliance.is_finite() || compliance < 0.0 {
            return false;
        }
        self.volume_conservation = Some(compliance);
        true
    }

    /// Phase 16: disables the dedicated volume-conservation constraint (`None`),
    /// reverting tetra volume to the global solver compliance.
    pub fn clear_volume_conservation(&mut self) {
        self.volume_conservation = None;
    }

    /// Phase 17: enables cohesion (adhesion / breakable glue) toward other bodies with
    /// the given `radius`, `stiffness` (compliance of the attraction constraint) and
    /// `break_distance` (separation at which a bond tears). Returns `false` (and does
    /// nothing) if `radius <= 0`, `stiffness < 0`, `break_distance <= radius`, or
    /// `break_distance`/`radius`/`stiffness` is `NaN`. `break_distance == +inf` is
    /// explicitly allowed (permanent, unbreakable glue).
    pub fn set_cohesion(&mut self, radius: Real, stiffness: Real, break_distance: Real) -> bool {
        if !radius.is_finite()
            || !stiffness.is_finite()
            || break_distance.is_nan()
            || !(radius > 0.0)
            || stiffness < 0.0
            || break_distance <= radius
        {
            return false;
        }
        self.cohesion = Some(CohesionParams {
            radius,
            stiffness,
            break_distance,
        });
        true
    }

    /// Phase 17: disables cohesion (`None`).
    pub fn clear_cohesion(&mut self) {
        self.cohesion = None;
    }

    /// Phase 18: sets the global internal (structural) damping coefficient `d`.
    /// Each step every free particle's velocity is scaled by `1 − d`. `0` = no
    /// damping; values in `[0, 1)` settle oscillation faster; `d >= 1` would fully
    /// freeze motion and is rejected (returns `false`). Non-finite `d` is rejected.
    pub fn set_damping(&mut self, d: Real) -> bool {
        if !d.is_finite() || d < 0.0 || d >= 1.0 {
            return false;
        }
        self.damping = d;
        true
    }

    /// Phase 24: set the number of solver substeps per [`SoftBody::step`] call.
    /// `n >= 1` splits the frame `dt` into `n` equal slices, projecting constraints
    /// at a finer time resolution. A value of `0` is rejected (kept at the previous
    /// setting) so a body never silently degrades to a no-op step. See the
    /// `substeps` field for the stability/convergence rationale.
    pub fn set_substeps(&mut self, n: u32) -> bool {
        if n == 0 {
            return false;
        }
        self.substeps = n;
        true
    }

    /// Phase 12: broad-phase + projection for self-collision. Builds a uniform
    /// spatial hash (cell size `2·radius`) over the *free* particle positions,
    /// finds all pairs within `2·radius` that are NOT direct structural neighbours
    /// (linked by an existing distance constraint), and projects each apart as a
    /// stiff XPBD distance constraint (`rest = 2·radius`, `compliance = stiffness`).
    ///
    /// `alpha` is the XPBD `α̃ = compliance / dt²` already used by the caller. The
    /// caller decides how many times to invoke this (once per iteration in XPBD,
    /// a few times after integration in MassSpring). Pure positional projection:
    /// `self.particles` is mutated directly; no new solver mechanics.
    fn damp_contact_velocity_split(
        pa: &mut [SoftParticle],
        pb: &mut [SoftParticle],
        i: usize,
        j: usize,
        mu: Real,
    ) {
        let wi = pa[i].inv_mass;
        let wj = pb[j].inv_mass;
        let wsum = wi + wj;
        if wsum == 0.0 {
            return;
        }
        let delta = pa[i].pos - pb[j].pos;
        let len = delta.length();
        if len < 1e-12 {
            return;
        }
        let n = delta / len;
        let v_rel = pa[i].vel - pb[j].vel;
        let vn = v_rel.dot(n);
        let v_t = v_rel - n * vn;
        let corr = v_t * mu;
        pa[i].vel -= corr * (wi / wsum);
        pb[j].vel += corr * (wj / wsum);
    }

    fn damp_contact_velocity(ps: &mut [SoftParticle], i: usize, j: usize, mu: Real) {
        let wi = ps[i].inv_mass;
        let wj = ps[j].inv_mass;
        let wsum = wi + wj;
        if wsum == 0.0 {
            return;
        }
        let delta = ps[i].pos - ps[j].pos;
        let len = delta.length();
        if len < 1e-12 {
            return;
        }
        let n = delta / len; // contact normal (b → a)
        let v_rel = ps[i].vel - ps[j].vel;
        let vn = v_rel.dot(n);
        let v_t = v_rel - n * vn; // tangential relative velocity
        // Apply -μ·v_t distributed by inverse mass (like a velocity constraint).
        let corr = v_t * mu;
        ps[i].vel -= corr * (wi / wsum);
        ps[j].vel += corr * (wj / wsum);
    }

    fn solve_self_collisions(&mut self, alpha: Real) -> Vec<(usize, usize)> {
        let params = match self.self_collision {
            Some(p) => p,
            None => return Vec::new(),
        };
        let d = params.radius * 2.0;
        // Inverse-mass view aligned to particle order (read by the projection helper).
        let w: Vec<Real> = self.particles.iter().map(|p| p.inv_mass).collect();
        // Structural neighbour set: (min,max) of every distance-constraint edge.
        let mut neighbours: HashSet<(usize, usize)> = HashSet::new();
        for c in &self.distance_constraints {
            let (a, b) = (c.a, c.b);
            let key = if a <= b { (a, b) } else { (b, a) };
            neighbours.insert(key);
        }
        // Build spatial hash of free-particle indices.
        let cell = d;
        let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
        let mut free: Vec<usize> = Vec::new();
        for (i, p) in self.particles.iter().enumerate() {
            if p.inv_mass == 0.0 {
                continue; // pinned / bound particles don't self-collide
            }
            free.push(i);
            let key = (
                (p.pos.x / cell).floor() as i64,
                (p.pos.y / cell).floor() as i64,
                (p.pos.z / cell).floor() as i64,
            );
            grid.entry(key).or_default().push(i);
        }
        // For each free particle, test against its cell + 26 neighbours.
        let mut contacts: Vec<(usize, usize)> = Vec::new();
        for &i in &free {
            let pi = self.particles[i].pos;
            let ci = (
                (pi.x / cell).floor() as i64,
                (pi.y / cell).floor() as i64,
                (pi.z / cell).floor() as i64,
            );
            for gx in ci.0 - 1..=ci.0 + 1 {
                for gy in ci.1 - 1..=ci.1 + 1 {
                    for gz in ci.2 - 1..=ci.2 + 1 {
                        if let Some(bucket) = grid.get(&(gx, gy, gz)) {
                            for &j in bucket {
                                if j <= i {
                                    continue; // each unordered pair once
                                }
                                let key = if i <= j { (i, j) } else { (j, i) };
                                if neighbours.contains(&key) {
                                    continue; // structural link, not a collision
                                }
                                // Project apart using the shared XPBD primitive.
                                solve_distance_constraint(
                                    &mut self.particles,
                                    i,
                                    j,
                                    d,
                                    alpha,
                                    &w,
                                    &mut 0.0,
                                );
                                contacts.push((i, j));
                            }
                        }
                    }
                }
            }
        }
        contacts
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
        // Phase 11: internal pressure (balloon model) — computed once, added per particle.
        let pressure = self.pressure_forces();
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
            // Phase 11: internal pressure pushes free particles outward along the
            // surface normal (balloon model).
            p.force += pressure[i];
            p.force += spring[i];
        }
    }

    /// Advances velocities then positions by `dt` (semi-implicit Euler).
    /// Pinned particles (`inv_mass == 0`) are not moved. Phase 18: a global internal
    /// damping factor `1 − self.damping` is applied to each free particle's velocity.
    pub fn integrate(&mut self, dt: Real) {
        let keep = 1.0 - self.damping;
        for p in &mut self.particles {
            if p.inv_mass == 0.0 {
                continue;
            }
            // v += dt · M⁻¹ · f   (M⁻¹ = inv_mass for a point mass)
            p.vel += dir_scaled(p.force, dt * p.inv_mass);
            // Phase 18: global internal damping (skipped when damping == 0).
            if keep < 1.0 {
                p.vel *= keep;
            }
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
        // Phase 10: freeze over-yield deformation into rest lengths before
        // integrating, so elastic edges become permanently deformed. (No-op unless
        // `plasticity` is `Some`.)
        self.apply_plasticity();
        // Phase 5f: when collision coupling is on, the integration layer drives
        // particle positions/velocities from proxy rigid bodies (after the
        // rigid-body narrow-phase/contact step), so we must not integrate here.
        if self.collide {
            return;
        }
        // Phase 24: subdivide the frame into `substeps` equal slices and run the
        // active solver once per slice. Each call to `step_xpbd` / `step_mass_spring`
        // resets its own Lagrange accumulators, so looping here gives independent
        // projection at a finer time resolution without touching the solver internals.
        // `substeps` is clamped to ≥1 so a 0 (or unset) value reproduces the single
        // substep behaviour.
        let n_sub = self.substeps.max(1) as usize;
        let sub_dt = dt / n_sub as Real;
        for _ in 0..n_sub {
            match self.solver {
                SoftSolver::MassSpring => self.step_mass_spring(sub_dt),
                SoftSolver::Xpbd { .. } => self.step_xpbd(sub_dt),
            }
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
        // Phase 12: self-collision as positional projection (a handful of passes).
        // Reuses the same broad-phase + XPBD push-apart as the XPBD path; the
        // compliance comes from `self.self_collision.stiffness`.
        if self.self_collision.is_some() {
            let stiffness = self.self_collision.unwrap().stiffness;
            let mu = self.self_collision.unwrap().friction;
            let alpha = stiffness / (dt * dt);
            let mut all_contacts: Vec<(usize, usize)> = Vec::new();
            for _ in 0..4 {
                all_contacts.extend(self.solve_self_collisions(alpha));
            }
            // Phase 20: velocity-level friction (vel is valid post-integrate).
            if let Some(mu) = mu {
                for (i, j) in all_contacts {
                    Self::damp_contact_velocity(&mut self.particles, i, j, mu);
                }
            }
        }
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
            compression: compliance,
        });
        Some(idx)
    }

    /// Phase 13 — sets the `stiffness` (Hookean `k`) of an existing spring (by the
    /// index returned from `add_spring`) at runtime. Lets callers tune a body's
    /// material heterogeneity after construction (e.g. stiffen a "bone", loosen a
    /// "tendon") without rebuilding the topology. Returns `false` for an out-of-range
    /// index or a negative/non-finite stiffness.
    pub fn set_spring_stiffness(&mut self, index: usize, stiffness: Real) -> bool {
        if stiffness < 0.0 || !stiffness.is_finite() {
            return false;
        }
        match self.springs.get_mut(index) {
            Some(s) => {
                s.stiffness = stiffness;
                true
            }
            None => false,
        }
    }

    /// Phase 13 — sets the XPBD `compliance` (α) of an existing distance constraint
    /// (by the index returned from `add_distance_constraint`) at runtime. Per-constraint
    /// compliance is honored by the XPBD solver (see `step_xpbd`). Phase 19: this sets
    /// **both** the stretch and compression compliance to the same value, i.e. it keeps
    /// the edge isotropic. Use `set_distance_constraint_compression` to make it
    /// anisotropic (different stretch vs compression softness). Returns `false` for
    /// an out-of-range index or a negative/non-finite compliance.
    pub fn set_distance_constraint_compliance(&mut self, index: usize, compliance: Real) -> bool {
        if compliance < 0.0 || !compliance.is_finite() {
            return false;
        }
        match self.distance_constraints.get_mut(index) {
            Some(c) => {
                c.compliance = compliance;
                c.compression = compliance;
                true
            }
            None => false,
        }
    }

    /// Phase 19 — sets the XPBD **compression** compliance `α_c` of an existing
    /// distance constraint (by the index returned from `add_distance_constraint`)
    /// at runtime, independently of its stretch compliance. This is the anisotropic
    /// knob: a cloth edge can resist stretch (`compliance`, small) but fold/compress
    /// easily (`compression`, large). The solver selects the compliance by the
    /// current strain sign each iteration (see `step_xpbd`). Returns `false` for an
    /// out-of-range index or a negative/non-finite compliance.
    pub fn set_distance_constraint_compression(&mut self, index: usize, compression: Real) -> bool {
        if compression < 0.0 || !compression.is_finite() {
            return false;
        }
        match self.distance_constraints.get_mut(index) {
            Some(c) => {
                c.compression = compression;
                true
            }
            None => false,
        }
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

    /// Phase 21 — adaptive tetrahedral subdivision (1 → 4 barycentric split).
    ///
    /// Each source tetrahedron `[a,b,c,d]` gains one new particle at its centroid
    /// (position = vertex mean; `inv_mass` = mean of the four endpoints, so a centroid
    /// bounded by pinned vertices stays effectively pinned) and is replaced by four
    /// sub-tetrahedra sharing that centroid: `(m,a,b,c)`, `(m,a,b,d)`, `(m,a,c,d)`,
    /// `(m,b,c,d)`. The four sub-volumes sum exactly to the parent volume, so the
    /// XPBD volume-conservation constraint (Phase 16) stays consistent — the centroid
    /// particle is a vertex of every sub-tet, so it is driven by those volume
    /// constraints directly (no extra distance edges are added, which would
    /// over-constrain and destabilise the solve). The shell topology (`triangles`)
    /// is left untouched — this is a volumetric refinement.
    ///
    /// *Adaptive*: a source tet is only split when its longest edge exceeds
    /// `max_edge_len`. Pass `max_edge_len = +∞` (the default when `!max_edge_len.is_finite()`)
    /// to subdivide every tet unconditionally. Returns the number of source tetrahedra
    /// actually split (0 if none qualified, e.g. all edges already short enough).
    ///
    /// Pure topology edit — no solver state, no SoA interaction. Determinism: source
    /// tets are processed in index order, so subdivision is reproducible.
    pub fn subdivide_tetrahedra(&mut self, max_edge_len: Real) -> usize {
        let src_tets: Vec<[u32; 4]> = self.tetrahedra.clone();
        let src_rests: Vec<Real> = self.tetra_rest_volumes.clone();
        if src_tets.is_empty() {
            return 0;
        }
        // Adaptive filter: only split tets whose longest edge exceeds the threshold.
        let adaptive = max_edge_len.is_finite();
        let mut new_tets: Vec<[u32; 4]> = Vec::with_capacity(src_tets.len() * 4);
        let mut new_rests: Vec<Real> = Vec::with_capacity(src_tets.len() * 4);
        let mut split_count = 0usize;
        for (ti, &tet) in src_tets.iter().enumerate() {
            let [a, b, c, d] = tet;
            let (pa, pb, pc, pd) = (
                self.particles[a as usize].pos,
                self.particles[b as usize].pos,
                self.particles[c as usize].pos,
                self.particles[d as usize].pos,
            );
            let longest = (pa - pb)
                .length()
                .max((pa - pc).length())
                .max((pa - pd).length())
                .max((pb - pc).length())
                .max((pb - pd).length())
                .max((pc - pd).length());
            if adaptive && longest <= max_edge_len {
                // Keep the parent tet unchanged.
                new_tets.push(tet);
                new_rests.push(src_rests[ti]);
                continue;
            }
            // Centroid particle (mean position + mean inverse mass).
            let mpos = (pa + pb + pc + pd) * 0.25;
            let im = (self.particles[a as usize].inv_mass
                + self.particles[b as usize].inv_mass
                + self.particles[c as usize].inv_mass
                + self.particles[d as usize].inv_mass)
                * 0.25;
            let m = self.particles.len() as u32;
            self.particles.push(SoftParticle {
                pos: mpos,
                vel: Vector::ZERO,
                force: Vector::ZERO,
                inv_mass: im,
                bound_body: None,
                bound_local: Vector::ZERO,
            });
            // Four sub-tetrahedra; each sub-rest-volume = 1/4 of the parent.
            let sub_rest = src_rests[ti] * 0.25;
            for sub in [[m, a, b, c], [m, a, b, d], [m, a, c, d], [m, b, c, d]] {
                new_tets.push(sub);
                new_rests.push(sub_rest);
            }
            split_count += 1;
        }
        self.tetrahedra = new_tets;
        self.tetra_rest_volumes = new_rests;
        split_count
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
                    compression: 0.0,
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
            compression: 0.0,
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

    /// Phase 10: enables/disables plasticity. Pass `None` to disable (perfectly
    /// elastic, the default). With `Some(PlasticityParams { yield_strain, creep })`,
    /// edges whose elastic strain magnitude exceeds `yield_strain` permanently shift
    /// their rest length toward the current length by `creep` (clamped to `[0,1]`)
    /// each step — the deformation freezes in instead of springing back.
    ///
    /// `yield_strain` is clamped to `≥ 0`; `creep` is clamped to `[0,1]`.
    pub fn set_plasticity(&mut self, params: Option<PlasticityParams>) {
        self.plasticity = params.map(|p| PlasticityParams {
            yield_strain: p.yield_strain.max(0.0),
            creep: p.creep.clamp(0.0, 1.0),
        });
    }

    /// Phase 10: permanently deforms over-yielded edges. Called once at the top of
    /// [`SoftBody::step`] after [`SoftBody::tear`]; a no-op when `plasticity` is
    /// `None`.
    ///
    /// For every structural edge (XPBD distance constraint / MassSpring spring) the
    /// elastic strain `s = (|len| − rest) / rest` is measured. If `|s| > yield_strain`,
    /// the over-yield portion is frozen: `rest += creep · (len − rest)` (the rest
    /// length moves toward the current length, so the edge no longer pulls back toward
    /// its original size). Pure rest-length edit — neither the SoA solver nor the
    /// integration order is touched.
    pub fn apply_plasticity(&mut self) {
        let (yield_strain, creep) = match self.plasticity {
            Some(p) => (p.yield_strain, p.creep),
            None => return,
        };
        if yield_strain <= 0.0 || creep <= 0.0 {
            return; // degenerate: no plasticity.
        }

        // Distance constraints (XPBD path). Filtered first into a `Vec`, then the
        // inner `self.particles` borrow for length measurement is released before we
        // mutate `self.distance_constraints` — keeps the two borrows disjoint.
        let new_rests: Vec<Real> = self
            .distance_constraints
            .iter()
            .map(|c| {
                let (pa, pb) = match (self.particles.get(c.a), self.particles.get(c.b)) {
                    (Some(a), Some(b)) => (a, b),
                    _ => return c.rest, // dangling edge → leave unchanged.
                };
                let len = (pb.pos - pa.pos).length();
                if c.rest <= 0.0 {
                    return c.rest;
                }
                let strain = (len - c.rest) / c.rest;
                if strain.abs() <= yield_strain {
                    c.rest
                } else {
                    // Freeze part of the elastic stretch into the rest length.
                    c.rest + creep * (len - c.rest)
                }
            })
            .collect();
        for (c, nr) in self.distance_constraints.iter_mut().zip(new_rests) {
            c.rest = nr;
        }

        // MassSpring springs.
        let new_springs: Vec<Spring> = self
            .springs
            .iter()
            .map(|s| {
                let (pa, pb) = match (self.particles.get(s.a), self.particles.get(s.b)) {
                    (Some(a), Some(b)) => (a, b),
                    _ => return *s,
                };
                let len = (pb.pos - pa.pos).length();
                if s.rest_length <= 0.0 {
                    return *s;
                }
                let strain = (len - s.rest_length) / s.rest_length;
                if strain.abs() <= yield_strain {
                    *s
                } else {
                    let mut ns = *s;
                    ns.rest_length += creep * (len - s.rest_length);
                    ns
                }
            })
            .collect();
        self.springs = new_springs;
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
        // Phase 11: internal pressure (balloon model) as a pure external
        // acceleration, applied alongside gravity/wind in the predict step.
        let pressure_forces = self.pressure_forces();
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
            // Phase 11: pressure force → acceleration (a += M⁻¹ · F).
            a += pressure_forces[i] * p.inv_mass;
            p.vel += dir_scaled(a, dt);
            p.pos += dir_scaled(p.vel, dt);
        }

        // 2. Project (fixed order → deterministic).
        // Extract constraint parameters into local buffers so the projection loop
        // can mutate `self.particles` through `&mut self` without holding an
        // immutable borrow of `self.distance_constraints` / `self.tetrahedra`
        // (avoids the borrow checker's simultaneous mutable+immutable error).
        // Order is preserved → determinism is unchanged.
        let ndc = self.distance_constraints.len();
        let mut d_a = Vec::with_capacity(ndc);
        let mut d_b = Vec::with_capacity(ndc);
        let mut d_rest = Vec::with_capacity(ndc);
        let mut d_comp = Vec::with_capacity(ndc);
        let mut d_compress = Vec::with_capacity(ndc);
        for c in &self.distance_constraints {
            d_a.push(c.a);
            d_b.push(c.b);
            d_rest.push(c.rest);
            d_comp.push(c.compliance);
            d_compress.push(c.compression);
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
        // Body-wide alpha for volume constraints. Phase 16: a dedicated
        // `volume_conservation` compliance overrides the solver's default, so the
        // tetra volume can be held hard (incompressible) even when the distance
        // solver is soft. Falls back to the solver compliance when unset.
        let vol_alpha = if let Some(c) = self.volume_conservation {
            c / (dt * dt)
        } else {
            self.xpbd_alpha(dt)
        };
        // Self-collision repulsion uses the compliance from `self_collision.stiffness`.
        let sc_alpha = if let Some(p) = self.self_collision {
            p.stiffness / (dt * dt)
        } else {
            0.0
        };
        // Phase 20: accumulate self-collision contacts across iterations for
        // post-recovery tangential friction.
        let mut self_contacts: Vec<(usize, usize)> = Vec::new();
        for _ in 0..iterations {
            for ci in 0..ndc {
                // Phase 13: honor per-constraint compliance → per-constraint α̃.
                // Phase 19: pick stretch vs compression compliance by the current
                // strain sign (tension uses `compliance`, compression uses
                // `compression`), enabling anisotropic edges.
                let len = (self.particles[d_a[ci]].pos - self.particles[d_b[ci]].pos).length();
                let c_alpha = if len > d_rest[ci] {
                    d_comp[ci]
                } else {
                    d_compress[ci]
                } / (dt * dt);
                solve_distance_constraint(
                    &mut self.particles,
                    d_a[ci],
                    d_b[ci],
                    d_rest[ci],
                    c_alpha,
                    &w,
                    &mut d_lambda[ci],
                );
            }
            for ti in 0..ntet {
                solve_volume_constraint(
                    &mut self.particles,
                    &t_idx[ti],
                    t_rest[ti],
                    vol_alpha,
                    &w,
                    &mut v_lambda[ti],
                );
            }
            // Phase 12: self-collision projection (broad-phase + push-apart) once
            // per iteration, interleaved with the structural constraints. Phase 20:
            // contacts are accumulated so tangential friction can be applied after
            // velocity recovery (where velocities are valid).
            self_contacts.extend(self.solve_self_collisions(sc_alpha));
        }

        // 3. Recover velocities.
        let keep = 1.0 - self.damping;
        for (i, p) in self.particles.iter_mut().enumerate() {
            if w[i] == 0.0 {
                continue;
            }
            p.vel = (p.pos - prev[i]) / dt;
            // Phase 18: global internal damping — bleed a fixed fraction of velocity
            // each step (jelly / slime energy loss). Skipped when damping == 0.
            if keep < 1.0 {
                p.vel *= keep;
            }
        }
        // Phase 20: velocity-level Coulomb friction at every self-collision contact.
        // Runs after velocity recovery so `vel` reflects the post-projection motion.
        if let Some(mu) = self.self_collision.and_then(|p| p.friction) {
            for (i, j) in self_contacts {
                Self::damp_contact_velocity(&mut self.particles, i, j, mu);
            }
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

/// Phase 14 — world-level soft-soft (cross-body) collision.
///
/// Runs after every soft body has been stepped. For each unordered pair of bodies
/// `(a, b)` with `a < b` that *both* have `cross_collision` set, it builds a uniform
/// spatial hash over the free particles of **both** bodies (cell size `2·R`, where
/// `R = min(radius_a, radius_b)`), finds all inter-body particle pairs whose centres
/// are within `2·R`, and pushes them apart with the same XPBD distance projection used
/// by self-collision (`rest = 2·R`, compliance = `min(stiffness_a, stiffness_b)`).
///
/// Pairs are projected with a few Gauss-Seidel iterations for stability. Body ids are
/// enumerated in ascending `SoftBodyId` order and particle pairs in index order, so the
/// result is deterministic. Pinned particles (`inv_mass == 0`) and same-body pairs are
/// skipped. This is pure positional projection — it adds no forces and does not touch
/// the SoA solver.
pub fn solve_cross_body_collisions(set: &mut SoftBodySet, dt: Real) {
    // Collect ids of bodies with cross-collision enabled, ascending by inner u32.
    let mut ids: Vec<SoftBodyId> = Vec::new();
    for (id, sb) in set.iter() {
        if sb.cross_collision.is_some() {
            ids.push(id);
        }
    }
    ids.sort_by_key(|id| id.0);
    let n = ids.len();
    // A few iterations of inter-body projection for stability.
    for _iter in 0..3 {
        for ia in 0..n {
            for ib in (ia + 1)..n {
                let a = ids[ia];
                let b = ids[ib];
                let (pa, pb) = match (set.get(a), set.get(b)) {
                    (Some(x), Some(y)) => (x, y),
                    _ => continue,
                };
                let (ca, cb) = match (pa.cross_collision, pb.cross_collision) {
                    (Some(x), Some(y)) => (x, y),
                    _ => continue,
                };
                let radius = ca.radius.min(cb.radius);
                let stiffness = ca.stiffness.min(cb.stiffness);
                // Phase 20: effective contact friction = min of both bodies' μ (a frictionless
                // body in the pair makes the contact frictionless, like real Coulomb coupling).
                let friction = match (ca.friction, cb.friction) {
                    (Some(x), Some(y)) => Some(x.min(y)),
                    _ => None,
                };
                let d = radius * 2.0;
                let compliance = stiffness / (dt * dt);
                let mut pos_a: Vec<Vector> = Vec::new();
                let mut w_a: Vec<Real> = Vec::new();
                let mut map_a: HashMap<usize, usize> = HashMap::new();
                for (oi, p) in pa.particles.iter().enumerate() {
                    if p.inv_mass != 0.0 {
                        map_a.insert(oi, pos_a.len());
                        pos_a.push(p.pos);
                        w_a.push(p.inv_mass);
                    }
                }
                let mut pos_b: Vec<Vector> = Vec::new();
                let mut w_b: Vec<Real> = Vec::new();
                let mut map_b: HashMap<usize, usize> = HashMap::new();
                for (oi, p) in pb.particles.iter().enumerate() {
                    if p.inv_mass != 0.0 {
                        map_b.insert(oi, pos_b.len());
                        pos_b.push(p.pos);
                        w_b.push(p.inv_mass);
                    }
                }
                let cell = d;
                let mut grid: HashMap<(i64, i64, i64), Vec<(usize, bool)>> = HashMap::new();
                for (si, p) in pos_a.iter().enumerate() {
                    let key = (
                        (p.x / cell).floor() as i64,
                        (p.y / cell).floor() as i64,
                        (p.z / cell).floor() as i64,
                    );
                    grid.entry(key).or_insert_with(Vec::new).push((si, false));
                }
                for (si, p) in pos_b.iter().enumerate() {
                    let key = (
                        (p.x / cell).floor() as i64,
                        (p.y / cell).floor() as i64,
                        (p.z / cell).floor() as i64,
                    );
                    grid.entry(key).or_insert_with(Vec::new).push((si, true));
                }
                let mut pairs: Vec<(usize, usize)> = Vec::new();
                for (si, p) in pos_a.iter().enumerate() {
                    let cx = (p.x / cell).floor() as i64;
                    let cy = (p.y / cell).floor() as i64;
                    let cz = (p.z / cell).floor() as i64;
                    for ox in -1..=1i64 {
                        for oy in -1..=1i64 {
                            for oz in -1..=1i64 {
                                if let Some(bucket) = grid.get(&(cx + ox, cy + oy, cz + oz)) {
                                    for &(sj, is_b) in bucket {
                                        if !is_b {
                                            continue;
                                        }
                                        let delta = pos_a[si] - pos_b[sj];
                                        let dist = delta.length();
                                        if dist < d && dist > 1e-9 {
                                            pairs.push((si, sj));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                for &(sa, sb_idx) in &pairs {
                    let (body_a, body_b) = set.get2_mut(a, b);
                    // Resolve original particle indices from the slot maps.
                    let oa_idx = map_a.iter().find(|(_, v)| **v == sa).map(|(&k, _)| k);
                    let ob_idx = map_b.iter().find(|(_, v)| **v == sb_idx).map(|(&k, _)| k);
                    if let (Some(oa_idx), Some(ob_idx)) = (oa_idx, ob_idx) {
                        if let (Some(ba), Some(bb)) = (body_a, body_b) {
                            let wa = w_a[sa];
                            let wb = w_b[sb_idx];
                            let pa_now = ba.particles[oa_idx].pos;
                            let pb_now = bb.particles[ob_idx].pos;
                            let delta = pa_now - pb_now;
                            let dist = delta.length();
                            if dist < d && dist > 1e-9 {
                                let n = delta / dist;
                                let cval = dist - d;
                                let wsum = wa + wb;
                                if wsum > 0.0 {
                                    let dlambda = (-cval - compliance * 0.0) / (wsum + compliance);
                                    if wa != 0.0 {
                                        ba.particles[oa_idx].pos += dir_scaled(n, wa * dlambda);
                                    }
                                    if wb != 0.0 {
                                        bb.particles[ob_idx].pos -= dir_scaled(n, wb * dlambda);
                                    }
                                    // Phase 20: velocity-level Coulomb friction on the
                                    // tangential relative slip (velocities are valid here,
                                    // post step_xpbd).
                                    if let Some(mu) = friction {
                                        SoftBody::damp_contact_velocity_split(
                                            &mut ba.particles,
                                            &mut bb.particles,
                                            oa_idx,
                                            ob_idx,
                                            mu,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Phase 17: world-level cohesion (adhesion / breakable glue) between soft bodies.
///
/// For every pair of bodies that both have `cohesion` set, free particles of `a` within
/// `radius = min(radius_a, radius_b)` of a free particle of `b` are attracted toward
/// contact (rest distance `radius`) via an XPBD constraint with compliance
/// `min(stiffness_a, stiffness_b) / dt²`. This bonds the two bodies together like glue —
/// the dual of Phase 9 tearing (which *breaks* edges; this *creates* bonds between bodies).
///
/// Bonds are *breakable*: if the pair is already separated by more than
/// `min(break_distance_a, break_distance_b)` the attraction is skipped (the glue has torn
/// and is not re-formed this step). Because the solve is stateless and re-evaluated every
/// step from current positions, a pair that drifts back within `radius` re-bonds — unless
/// `break_distance` is `inf` (permanent glue). Runs a few iterations for stability. Shares
/// the spatial-hash structure of [`solve_cross_body_collisions`]. Pure positional projection
/// — no new solver mechanics, no SoA interaction.
pub fn solve_cohesion(set: &mut SoftBodySet, dt: Real) {
    let mut ids: Vec<SoftBodyId> = Vec::new();
    for (id, sb) in set.iter() {
        if sb.cohesion.is_some() {
            ids.push(id);
        }
    }
    ids.sort_by_key(|id| id.0);
    let n = ids.len();
    for _iter in 0..3 {
        for ia in 0..n {
            for ib in (ia + 1)..n {
                let a = ids[ia];
                let b = ids[ib];
                let (pa, pb) = match (set.get(a), set.get(b)) {
                    (Some(x), Some(y)) => (x, y),
                    _ => continue,
                };
                let (ca, cb) = match (pa.cohesion, pb.cohesion) {
                    (Some(x), Some(y)) => (x, y),
                    _ => continue,
                };
                let radius = ca.radius.min(cb.radius);
                let stiffness = ca.stiffness.min(cb.stiffness);
                let break_distance = ca.break_distance.min(cb.break_distance);
                // `capture` = how far apart two free particles may be and still form a bond
                // (must exceed the rest distance `radius`, else indistinguishable from a
                // non-overlapping contact). `d` = rest distance of the attraction (contact).
                let capture = radius * 2.0;
                let d = radius;
                let compliance = stiffness / (dt * dt);
                let mut pos_a: Vec<Vector> = Vec::new();
                let mut w_a: Vec<Real> = Vec::new();
                let mut map_a: HashMap<usize, usize> = HashMap::new();
                for (oi, p) in pa.particles.iter().enumerate() {
                    if p.inv_mass != 0.0 {
                        map_a.insert(oi, pos_a.len());
                        pos_a.push(p.pos);
                        w_a.push(p.inv_mass);
                    }
                }
                let mut pos_b: Vec<Vector> = Vec::new();
                let mut w_b: Vec<Real> = Vec::new();
                let mut map_b: HashMap<usize, usize> = HashMap::new();
                for (oi, p) in pb.particles.iter().enumerate() {
                    if p.inv_mass != 0.0 {
                        map_b.insert(oi, pos_b.len());
                        pos_b.push(p.pos);
                        w_b.push(p.inv_mass);
                    }
                }
                let cell = capture;
                let mut grid: HashMap<(i64, i64, i64), Vec<(usize, bool)>> = HashMap::new();
                for (si, p) in pos_a.iter().enumerate() {
                    let key = (
                        (p.x / cell).floor() as i64,
                        (p.y / cell).floor() as i64,
                        (p.z / cell).floor() as i64,
                    );
                    grid.entry(key).or_insert_with(Vec::new).push((si, false));
                }
                for (si, p) in pos_b.iter().enumerate() {
                    let key = (
                        (p.x / cell).floor() as i64,
                        (p.y / cell).floor() as i64,
                        (p.z / cell).floor() as i64,
                    );
                    grid.entry(key).or_insert_with(Vec::new).push((si, true));
                }
                let mut pairs: Vec<(usize, usize)> = Vec::new();
                for (si, p) in pos_a.iter().enumerate() {
                    let cx = (p.x / cell).floor() as i64;
                    let cy = (p.y / cell).floor() as i64;
                    let cz = (p.z / cell).floor() as i64;
                    for ox in -1..=1i64 {
                        for oy in -1..=1i64 {
                            for oz in -1..=1i64 {
                                if let Some(bucket) = grid.get(&(cx + ox, cy + oy, cz + oz)) {
                                    for &(sj, is_b) in bucket {
                                        if !is_b {
                                            continue;
                                        }
                                        let delta = pos_a[si] - pos_b[sj];
                                        let dist = delta.length();
                                        // Bond only when within capture radius AND not
                                        // already torn apart beyond break_distance.
                                        if dist < capture && dist > 1e-9 && dist < break_distance {
                                            pairs.push((si, sj));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                for &(sa, sb_idx) in &pairs {
                    let (body_a, body_b) = set.get2_mut(a, b);
                    let oa_idx = map_a.iter().find(|(_, v)| **v == sa).map(|(&k, _)| k);
                    let ob_idx = map_b.iter().find(|(_, v)| **v == sb_idx).map(|(&k, _)| k);
                    if let (Some(oa_idx), Some(ob_idx)) = (oa_idx, ob_idx) {
                        if let (Some(ba), Some(bb)) = (body_a, body_b) {
                            let wa = w_a[sa];
                            let wb = w_b[sb_idx];
                            let pa_now = ba.particles[oa_idx].pos;
                            let pb_now = bb.particles[ob_idx].pos;
                            let delta = pa_now - pb_now;
                            let dist = delta.length();
                            if dist < capture && dist > 1e-9 && dist < break_distance {
                                let nrm = delta / dist;
                                // Attract: pull the two particles toward contact distance d.
                                // c(dist) = dist - d  (>0 means too far -> attract inward).
                                let cval = dist - d;
                                let wsum = wa + wb;
                                if wsum > 0.0 {
                                    let dlambda = (-cval - compliance * 0.0) / (wsum + compliance);
                                    if wa != 0.0 {
                                        ba.particles[oa_idx].pos += dir_scaled(nrm, wa * dlambda);
                                    }
                                    if wb != 0.0 {
                                        bb.particles[ob_idx].pos -= dir_scaled(nrm, wb * dlambda);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
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

    /// Iterator over all live `(SoftBodyId, &SoftBody)` pairs, in ascending id order.
    pub fn iter(&self) -> impl Iterator<Item = (SoftBodyId, &SoftBody)> {
        self.bodies
            .iter()
            .enumerate()
            .filter_map(|(i, b)| b.as_ref().map(|sb| (SoftBodyId(i as u32), sb)))
    }

    /// Simultaneous mutable access to two distinct live bodies. Returns
    /// `(None, None)` if either id is unknown/removed or the two ids are equal.
    /// Used by the Phase 14 cross-body collision pass to project two bodies apart.
    pub fn get2_mut(
        &mut self,
        a: SoftBodyId,
        b: SoftBodyId,
    ) -> (Option<&mut SoftBody>, Option<&mut SoftBody>) {
        if a == b {
            return (None, None);
        }
        let (lo, hi) = if a.0 <= b.0 { (a, b) } else { (b, a) };
        let (lo_i, hi_i) = (lo.0 as usize, hi.0 as usize);
        let len = self.bodies.len();
        if lo_i >= len || hi_i >= len {
            return (None, None);
        }
        // Split the slice so the two borrows are disjoint.
        let (left, right) = self.bodies.split_at_mut(hi_i);
        let first = left.get_mut(lo_i).and_then(|b| b.as_mut());
        let second = right.get_mut(0).and_then(|b| b.as_mut());
        if a.0 <= b.0 {
            (first, second)
        } else {
            (second, first)
        }
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
