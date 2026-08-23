//! Force containers — a kind-classified, self-integrating force model.
//!
//! ## Motivation
//!
//! Rapier's legacy force model is "flat": user forces (`user_force`) are
//! accumulated and then wiped every step by `reset_forces`, so a persistent
//! force (thrust, a magnetic anchor) must be re-`add_force`d every frame — a
//! tedious remove-then-re-apply ritual. Gravity, by contrast, is already a
//! *persistent* force (it is reapplied every step, never cleared).
//!
//! This module replaces that flat model with a **{persistent, transient}**
//! lifecycle expressed through **force containers classified by kind**:
//!
//! * Each force *kind* (gravity, thrust, magnetic, wind, friction, contact
//!   reaction, one-shot event, user, …) owns a [`KindContainer`].
//! * A container is **self-integrating**: it holds its own entries, its own
//!   [`ForceKind`], and — crucially — its own [`Persistence`] flag recorded
//!   *inside* the container (not via two separate struct types).
//! * Whether a force survives across steps is decided by the container's own
//!   [`ForceContainer::end_frame`] using its internal `persistence` field: a
//!   `Persistent` container (e.g. gravity, steady thrust) keeps its entries;
//!   a `Transient` container (e.g. friction, contact reaction, one-shot event)
//!   drains itself every step.
//!
//! Adding a new force kind = add a `ForceKind` variant + create a
//! `KindContainer` with the right `kind`/`persistence`. The effective-force
//! summation (`compute_body_effective_forces`) is transparent to kinds.
//!
//! ## Trait design
//!
//! * [`ForceContribution`] — one force entry. Provides a single generic
//!   [`ForceContribution::accumulate`] method that writes the entry into an
//!   [`EffectiveForce`] accumulator (including the `r × F` torque term).
//! * [`ForceContainer`] — a kind-classified, self-integrating holder of
//!   `ForceContribution`s. Knows its own `kind()` and `persistence()`, and
//!   implements frame-end cleanup (`end_frame`/`clear`).

use std::collections::HashMap;
use std::vec::Vec;

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use crate::dynamics::RigidBodySet;
use crate::dynamics::rigid_body::RigidBody;
use crate::geometry::NarrowPhase;
use crate::math::{AngVector, Real, Vector};
use crate::utils::CrossProduct;
use crate::utils::OrthonormalBasis;

/// Effective (already-summed) force + torque for one body, consumed by the solver.
#[derive(Clone, Copy, Debug, Default)]
pub struct EffectiveForce {
    /// Linear force (world frame).
    pub force: Vector,
    /// Torque / angular force (world frame).
    pub torque: AngVector,
}

/// Whether a container's forces survive across steps.
///
/// Recorded **inside** each [`KindContainer`] (see [`ForceContainer::persistence`]),
/// not as two separate struct types.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Persistence {
    /// Forces persist until explicitly removed (`remove`/`clear`).
    /// Example: gravity, steady thrust, magnetic anchor.
    Persistent,
    /// Forces are valid only for the current step and are drained at frame end.
    /// Example: one-shot events, wind gusts, contact friction / reaction.
    Transient,
}

/// The kind of a force, used to classify containers.
///
/// New force kinds are added here; a container simply stores the matching
/// `ForceKind` and a [`Persistence`] flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum ForceKind {
    /// World gravity (always persistent).
    Gravity,
    /// Steady / pulsed thrust.
    Thrust,
    /// Magnetic / electromagnetic field force.
    Magnetic,
    /// Aerodynamic / fluid wind.
    Wind,
    /// Surface friction (solver-emergent, transient).
    Friction,
    /// Contact normal reaction (solver-emergent, transient).
    ContactReaction,
    /// One-shot external event force.
    Event,
    /// Legacy `add_force` user force.
    User,
    /// User-defined custom kind.
    Custom(u32),
}

/// One force contribution inside a container.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct ForceEntry {
    /// Caller-managed id (so it can be removed individually).
    pub id: u64,
    /// Linear force component (world frame).
    pub force: Vector,
    /// Torque / angular force component (world frame).
    pub torque: AngVector,
    /// World-space application point; `None` = center of mass (no extra torque).
    pub point: Option<Vector>,
}

/// A single force contribution that can be summed into an [`EffectiveForce`].
pub trait ForceContribution {
    /// Linear force component (world frame).
    fn force(&self) -> Vector;
    /// Torque / angular force component (world frame).
    fn torque(&self) -> AngVector;
    /// World-space application point; `None` = center of mass.
    fn point(&self) -> Option<Vector>;

    /// Generic method: write this contribution into the accumulator.
    ///
    /// This is the single unified "store the force data" point — every force
    /// kind uses the same write path, including the `r × F` torque term from
    /// an off-center application point.
    fn accumulate(&self, eff: &mut EffectiveForce, world_com: Vector) {
        eff.force += self.force();
        eff.torque += self.torque();
        if let Some(p) = self.point() {
            eff.torque += (p - world_com).gcross(self.force());
        }
    }
}

impl ForceContribution for ForceEntry {
    fn force(&self) -> Vector {
        self.force
    }
    fn torque(&self) -> AngVector {
        self.torque
    }
    fn point(&self) -> Option<Vector> {
        self.point
    }
}

/// A self-integrating force container classified by [`ForceKind`].
///
/// The container holds its own entries, its own `kind`, and its own
/// [`Persistence`] flag (recorded inside — see [`persistence`](Self::persistence)).
/// Frame-end cleanup is delegated to the container via [`end_frame`](Self::end_frame).
pub trait ForceContainer {
    /// The force kind this container holds.
    fn kind(&self) -> ForceKind;
    /// Whether the contained forces persist across steps — **recorded inside the container**.
    fn persistence(&self) -> Persistence;
    /// Iterate the live contributions.
    fn contributions(&self) -> impl Iterator<Item = &dyn ForceContribution>;
    /// Remove one contribution by id. Returns `true` if removed.
    fn remove(&mut self, id: u64) -> bool;
    /// Clear all contributions (persistent containers usually shouldn't be cleared).
    fn clear(&mut self);

    /// Frame-end: self-integrating cleanup. Drains only if the container's
    /// internal `persistence` is `Transient`; `Persistent` containers keep their
    /// entries (e.g. gravity stays constant, no per-step re-apply needed).
    fn end_frame(&mut self) {
        if self.persistence() == Persistence::Transient {
            self.clear();
        }
    }
}

/// The concrete, kind-classified, self-integrating container.
///
/// A `GravityContainer`, `ThrustContainer`, `FrictionContainer`, … are all just
/// `KindContainer` instances differing in `kind` and `persistence`. This keeps
/// the container taxonomy flat and data-driven: adding a force kind is a data
/// change, not a new type.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct KindContainer {
    kind: ForceKind,
    persistence: Persistence,
    entries: Vec<ForceEntry>,
}

impl KindContainer {
    /// Create an empty container of the given kind and persistence.
    pub fn new(kind: ForceKind, persistence: Persistence) -> Self {
        Self {
            kind,
            persistence,
            entries: Vec::new(),
        }
    }

    /// Push a contribution, returning its id (auto-assigned if `id == 0`).
    pub fn push(&mut self, mut entry: ForceEntry) -> u64 {
        if entry.id == 0 {
            // Deterministic auto-id from current length + 1 (0 reserved as "auto").
            entry.id = (self.entries.len() as u64).wrapping_add(1);
            if entry.id == 0 {
                entry.id = 1;
            }
        }
        let id = entry.id;
        self.entries.push(entry);
        id
    }

    /// Number of live contributions.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the container is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl ForceContainer for KindContainer {
    fn kind(&self) -> ForceKind {
        self.kind
    }
    fn persistence(&self) -> Persistence {
        self.persistence
    }
    fn contributions(&self) -> impl Iterator<Item = &dyn ForceContribution> {
        self.entries.iter().map(|e| e as &dyn ForceContribution)
    }
    fn remove(&mut self, id: u64) -> bool {
        if let Some(pos) = self.entries.iter().position(|e| e.id == id) {
            self.entries.remove(pos);
            true
        } else {
            false
        }
    }
    fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Per-body map of force containers, classified by [`ForceKind`].
pub type BodyForceContainers = HashMap<ForceKind, KindContainer>;

/// Sum all force contributions of a body into its effective force/torque.
///
/// * Legacy `user_force`/`user_torque` are included for backward compatibility.
/// * World gravity is read from `gravity_container` (a `Persistent` container
///   whose entry stores the gravity *acceleration*; it is scaled by the body's
///   effective mass and `gravity_scale`, exactly like the legacy path).
/// * Every body container is summed; persistent and transient are treated
///   identically here — the lifecycle difference is resolved at frame end by
///   [`ForceContainer::end_frame`].
pub fn compute_body_effective_forces(rb: &mut RigidBody, gravity_container: &KindContainer) {
    let mut eff = EffectiveForce::default();

    // Legacy user slot (kept for backward compatibility).
    eff.force += rb.forces.user_force;
    eff.torque += rb.forces.user_torque;

    // World gravity (acceleration stored in the gravity container entries).
    let mass = rb.mprops.effective_mass();
    let gravity_scale = rb.forces.gravity_scale;
    for entry in &gravity_container.entries {
        eff.force += entry.force * mass * gravity_scale;
    }

    // Body containers, classified by kind. Persistence is not consulted here —
    // both persistent and transient forces act this step; the transient ones are
    // drained afterward by `drain_transient_forces`. Solver-emergent kinds
    // (`Friction`, `ContactReaction`) are *observation-only* readouts: the contact
    // solver already applies their impulses, so if any entry found its way into
    // these containers we must NOT re-sum them (that would double-apply contact
    // forces and destabilize the solve). The bridge that fills these containers
    // (`bridge_solver_contact_forces`) writes them but they are consumed only via
    // `force_container(ForceKind::Friction/ContactReaction)` queries, never here.
    let world_com = rb.mprops.world_com;
    for container in rb.force_containers.values() {
        if container.kind() == ForceKind::Friction || container.kind() == ForceKind::ContactReaction
        {
            continue;
        }
        for c in container.contributions() {
            c.accumulate(&mut eff, world_com);
        }
    }

    rb.forces.force = eff.force;
    rb.forces.torque = eff.torque;
}

/// Drain every transient (per-step) force container on a body.
///
/// Called at frame end so one-shot / event / contact forces do not leak into
/// the next step, while persistent containers (gravity, steady thrust) keep
/// their entries — eliminating the manual `reset_forces` ritual.
pub fn drain_transient_forces(rb: &mut RigidBody) {
    // No containers at all → nothing to drain. Early-return before taking any
    // mutable borrow so we never trip the body's "modified" flag during steady
    // state (which would force an island rebuild / active-set epoch bump).
    if rb.force_containers.is_empty() {
        return;
    }
    let mut drained_any = false;
    for container in rb.force_containers.values_mut() {
        if container.persistence() == Persistence::Transient && !container.entries.is_empty() {
            container.end_frame();
            drained_any = true;
        }
    }
    // If nothing was actually cleared, avoid leaving a dangling mutable borrow
    // that some callers interpret as a modification.
    let _ = drained_any;
}

/// Bridge the contact solver's *emergent* normal + friction impulses into the
/// `Friction` / `ContactReaction` force containers, as an **observation-only**
/// readout.
///
/// These forces are produced by the solver every step and already applied to the
/// bodies — they are NOT re-summed by [`compute_body_effective_forces`]. Writing
/// them into dedicated containers lets application code inspect the per-step
/// contact reaction / friction (e.g. for grip estimation, slip detection, or
/// `ContactForceEvent`-driven force injection) without disturbing the solve.
///
/// Each kind is rebuilt fresh every step (matching the `Transient` lifecycle:
/// the solver re-emerges them each step), so a sleeping pair that stops touching
/// simply leaves no entry next step — no manual drain needed.
///
/// # Determinism
///
/// Must run **after** `build_islands_and_solve_velocity_constraints` on a single
/// thread (the contact graph is not `Sync` and incremental graph maintenance
/// assumes no concurrent edge mutation). Callers (the pipeline) already hold the
/// whole `NarrowPhase` mutably here.
pub fn bridge_solver_contact_forces(
    narrow_phase: &NarrowPhase,
    bodies: &mut RigidBodySet,
    dt: Real,
) {
    let inv_dt = crate::utils::inv(dt);

    for pair in narrow_phase.contact_pairs() {
        // Reconstruct the per-pair world-space force by summing each contact's
        // normal impulse (reaction) and tangential impulse (friction), scaled by
        // inv_dt to convert impulse → force, and using the frozen lever arms
        // (solver_dp1) for the torque.
        let mut reaction = Vector::ZERO;
        let mut friction = Vector::ZERO;
        let mut r_torque = AngVector::default();
        let mut f_torque = AngVector::default();

        // Body handles come from each manifold's `ContactManifoldData` (a pair can
        // span multiple manifolds, but they share the same two bodies).
        let mut rb1 = None;
        let mut rb2 = None;

        for manifold in pair.solver_manifolds() {
            let normal = manifold.data.normal;
            let tangents = normal.orthonormal_basis();
            for c in manifold.contacts() {
                let jn = c.data.impulse * inv_dt;
                // Physical contact reaction force on `rigid_body1` = `-normal * jn`
                // (the contact pushes body1 *away* from body2; the solver's `impulse`
                // is positive for compression). Force on body2 is the opposite.
                let reaction_force = -normal * jn;
                reaction += reaction_force;
                // Friction force on body1 from the tangential impulse vector.
                let jt = c.data.tangent_impulse;
                #[cfg(feature = "dim2")]
                let friction_force = -tangents[0] * jt.x * inv_dt;
                #[cfg(feature = "dim3")]
                let friction_force = -(tangents[0] * jt.x + tangents[1] * jt.y) * inv_dt;
                friction += friction_force;

                // Torque = r × F using the frozen lever arm (solver_dp1).
                r_torque += c.data.solver_dp1.gcross(reaction_force);
                f_torque += c.data.solver_dp1.gcross(friction_force);
            }

            if rb1.is_none() {
                rb1 = manifold.data.rigid_body1;
                rb2 = manifold.data.rigid_body2;
            }
        }

        let (rb1, rb2) = match (rb1, rb2) {
            (Some(h1), Some(h2)) => (h1, h2),
            _ => continue,
        };

        // Body 1 gets +reaction / +friction; body 2 gets the opposite.
        if let Some(rb) = bodies.get_mut(rb1) {
            write_contact_observation(rb, reaction, friction, r_torque, f_torque);
        }
        if let Some(rb) = bodies.get_mut(rb2) {
            write_contact_observation(rb, -reaction, -friction, -r_torque, -f_torque);
        }
    }
}

/// Push one pair's contact reaction + friction into the body's observation-only
/// containers, rebuilding them each step (Transient lifecycle).
fn write_contact_observation(
    rb: &mut RigidBody,
    reaction: Vector,
    friction: Vector,
    r_torque: AngVector,
    f_torque: AngVector,
) {
    // Rebuild fresh: clear previous step's observation (it is solver-emergent,
    // so it re-emerges from scratch each step).
    rb.force_containers.insert(
        ForceKind::ContactReaction,
        KindContainer::new(ForceKind::ContactReaction, Persistence::Transient),
    );
    rb.force_containers.insert(
        ForceKind::Friction,
        KindContainer::new(ForceKind::Friction, Persistence::Transient),
    );

    if reaction.length_squared() > 0.0 {
        rb.force_containers
            .get_mut(&ForceKind::ContactReaction)
            .unwrap()
            .push(ForceEntry {
                id: 1,
                force: reaction,
                torque: r_torque,
                point: None,
            });
    }
    if friction.length_squared() > 0.0 {
        rb.force_containers
            .get_mut(&ForceKind::Friction)
            .unwrap()
            .push(ForceEntry {
                id: 1,
                force: friction,
                torque: f_torque,
                point: None,
            });
    }
}
