//! Structures related to dynamics: bodies, impulse_joints, etc.

#[cfg(feature = "alloc")]
pub use self::ccd::CCDSolver;
pub use self::coefficient_combine_rule::CoefficientCombineRule;
#[cfg(feature = "dim3")]
pub use self::integration_parameters::FrictionModel;
pub use self::integration_parameters::{IntegrationParameters, SpringCoefficients};
#[cfg(feature = "alloc")]
pub use self::island_manager::IslandManager;
#[cfg(feature = "alloc")]
pub(crate) use self::island_manager::{INVALID_ISLAND, ImpulseJointIslandEvent, PersistentIslands};

#[cfg(feature = "alloc")]
pub(crate) use self::joint::JointGraphEdge;
#[cfg(feature = "alloc")]
pub(crate) use self::joint::JointIndex;
pub use self::joint::*;
#[cfg(feature = "alloc")]
pub use self::rigid_body_components::*;
pub use self::rigid_body_handle::RigidBodyHandle;
#[cfg(feature = "alloc")]
pub(crate) use self::rigid_body_set::ModifiedRigidBodies;
#[cfg(feature = "alloc")]
pub(crate) use self::solver::StagedIslandSolver;
pub use parry::mass_properties::MassProperties;

#[cfg(feature = "alloc")]
pub use self::rigid_body::{RigidBody, RigidBodyBuilder};
#[cfg(feature = "alloc")]
pub use self::rigid_body_set::{BodyPair, RigidBodySet};

// Force containers: kind-classified, self-integrating {persistent, transient}
// force model. Declared AFTER `rigid_body` so the `RigidBody` type it depends
// on is already visible in this namespace.
#[cfg(feature = "alloc")]
pub use self::force_containers::*;

// Soft-body (deformable body) support — Phase 0 foundation. Independent of the
// SoA SIMD solver boundary; wires into `World` / `PersistentIslands` in a later
// phase (see `.hermes/plans/2026-08-24_soft-body-roadmap.md`).
#[cfg(feature = "alloc")]
pub mod soft_body;

#[cfg(feature = "alloc")]
mod ccd;
mod coefficient_combine_rule;
mod integration_parameters;
#[cfg(feature = "alloc")]
mod island_manager;
mod joint;
#[cfg(feature = "alloc")]
mod rigid_body_components;
mod rigid_body_handle;
#[cfg(feature = "alloc")]
pub(crate) mod solver;

#[cfg(feature = "alloc")]
mod rigid_body;
#[cfg(feature = "alloc")]
mod rigid_body_set;

#[cfg(feature = "alloc")]
pub mod force_containers;
