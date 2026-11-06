//! This fork's soft-body extensions, fused into the official `SoftBody`.
//!
//! Upstream v0.36 shipped a complete soft-body system (XPBD elements, clusters,
//! deformation-ready colliders, tearing that spawns bodies, FEM). This fork had
//! grown its own material/behaviour layer before that, and the merge left the two
//! side by side. This module is where the second one lands: the official body is
//! the only soft body, and the extra capabilities are stored here as data and
//! applied by the official solver instead of by a parallel integrator.
//!
//! ## What lives here (all of it is "ours", upstream has no equivalent)
//!
//! | 能力 | 数据 | 官方侧落点 |
//! | --- | --- | --- |
//! | 均匀风场 / 空气阻力 | [`Wind`] | 预测阶段对每个自由粒子加 `m·a − m·drag·v` |
//! | 内部气压（气球） | [`ForkExtras::pressure`] | 预测阶段沿外法线加 `P·A` |
//! | 体级内阻尼 | [`ForkExtras::damping`] | 速度恢复后按 `1 − damping` 缩放 |
//! | 温度场（热膨胀 + 软化） | [`ThermalParams`] | 材质 `young_modulus` 与静止长度的乘子 |
//! | 率相关粘弹性 | [`ViscoelasticParams`] | 边刚度乘 `1 + rate·|dε/dt|` |
//! | 体级正交各向异性 | [`ForkExtras::anisotropy`] | 边柔度除以 `nᵀ·diag·n` |
//! | 主动应变 / 肌纤维方向 | [`ForkExtras::activation`] / [`ForkExtras::fibre`] | 边静止长度 `rest·(1 − γ)`，沿纤维各向异性 |
//! | 内聚（可断裂胶水） | [`CohesionParams`] | 跨体粒子对的距离约束 |
//! | 自接触 / 跨体接触 | [`SelfCollisionParams`] | 官方 `self_contacts`，跨体走世界级接触对 |
//! | 撕裂准则（应变/应力/能量） | [`ForkTearCriterion`] | 官方 `tear_strain` / `tear_force` |
//! | 子步 | [`ForkExtras::substeps`] | 官方冲击自适应子步的额外细分 |
//!
//! Nothing in this module integrates anything by itself: every field is read by
//! the official solver path (`soft_constraint_prepare`, the per-particle predict,
//! and the velocity recovery), so there is exactly one soft-body implementation.

use crate::math::{Real, Vector};

/// A uniform wind / air-resistance field.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde-serialize",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct Wind {
    /// Constant wind acceleration (`m/s²`), applied to every free particle.
    pub accel: Vector,
    /// Linear air-resistance coefficient (`1/s`): `F_drag = −m·drag·v`.
    pub drag: Real,
}

/// Rate-dependent (viscoelastic) constitutive parameters.
///
/// The edge stiffness is multiplied by `1 + rate_coefficient·|dε/dt|`, so faster
/// deformation meets a stiffer material (shear-thickening). `0` is purely elastic.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(
    feature = "serde-serialize",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct ViscoelasticParams {
    /// Rate coefficient `η ≥ 0` (1/s).
    pub rate_coefficient: Real,
}

/// A uniform thermal field: thermal expansion plus temperature softening.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde-serialize",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct ThermalParams {
    /// Current temperature `T` (K).
    pub temp: Real,
    /// Reference temperature `T₀` (K): no expansion and no softening at `T = T₀`.
    pub ambient: Real,
    /// Linear thermal expansion coefficient `α` (1/K).
    pub expansion: Real,
    /// Stiffness-vs-temperature coefficient: `k_eff = k·(1 − coeff·(T − T₀))`.
    pub stiffness_temp_coeff: Real,
}

/// Self-contact parameters (particle spheres of `radius`).
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde-serialize",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct SelfCollisionParams {
    /// Particle collision radius.
    pub radius: Real,
    /// Contact stiffness (XPBD compliance-like, `0` = rigid).
    pub stiffness: Real,
    /// Coulomb friction coefficient `μ ∈ [0, 1]`.
    pub friction: Real,
}

/// Cohesion (adhesion / breakable glue) between the particles of two soft bodies.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde-serialize",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct CohesionParams {
    /// Interaction radius.
    pub radius: Real,
    /// Glue stiffness.
    pub stiffness: Real,
    /// Distance past which the glue breaks.
    pub break_distance: Real,
}

/// How an edge is judged torn, with its threshold.
///
/// Upstream only has a strain threshold and a force threshold; this is the fork's superset.
/// `Strain` and `Stress` map onto upstream's `material.tear_strain` / `material.tear_force`
/// (the setters write them there), while `Energy` has no upstream equivalent and is evaluated
/// by the solver as the work done over the current stretch.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde-serialize",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum ForkTearCriterion {
    /// Tear when `(|len| − rest) / rest` exceeds the threshold.
    Strain(Real),
    /// Tear when the edge force exceeds the threshold (newtons).
    Stress(Real),
    /// Tear when the elastic work `½·F·(|len| − rest)` exceeds the threshold (joules).
    Energy(Real),
}

/// This fork's soft-body extensions attached to one official [`SoftBody`](super::SoftBody).
///
/// Every field is optional and inert by default, so an official body that never
/// touches this struct behaves exactly like upstream's.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(
    feature = "serde-serialize",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct ForkExtras {
    /// Uniform wind field (see [`Wind`]).
    pub wind: Option<Wind>,
    /// Uniform internal pressure `P` (force/area), pushing along the outward normal.
    pub pressure: Option<Real>,
    /// Body-wide internal damping in `[0, 1)` (velocity scale `1 − damping` per step).
    pub damping: Real,
    /// Extra substeps for this body (`1` = the official adaptive substepping alone).
    pub substeps: u32,
    /// Uniform thermal field.
    pub thermal: Option<ThermalParams>,
    /// Rate-dependent stiffness.
    pub viscoelastic: Option<ViscoelasticParams>,
    /// Body-level orthotropic stiffness axes (each component ≥ 0).
    pub anisotropy: Option<Vector>,
    /// Cohesion toward other bodies that also set it.
    pub cohesion: Option<CohesionParams>,
    /// Self-contact.
    pub self_collision: Option<SelfCollisionParams>,
    /// Cross-body (soft-soft) contact, using the same parameters as self-contact.
    pub cross_collision: Option<SelfCollisionParams>,
    /// Tearing criterion (the threshold itself lives in the official
    /// `material.tear_strain` / `material.tear_force`).
    pub tear: Option<ForkTearCriterion>,
    /// Global active strain `γ ∈ [0, 1]`: edge rest lengths shrink to `rest·(1 − γ)`.
    pub activation: Real,
    /// Muscle-fibre direction: the active strain acts along it (anisotropic contraction).
    pub fibre: Option<Vector>,
}

impl ForkExtras {
    /// Whether every extension is at its inert default, so the official path can
    /// skip the whole fork layer for this body.
    pub fn is_inert(&self) -> bool {
        self.wind.is_none()
            && self.pressure.is_none()
            && self.damping == 0.0
            && self.substeps <= 1
            && self.thermal.is_none()
            && self.viscoelastic.is_none()
            && self.anisotropy.is_none()
            && self.cohesion.is_none()
            && self.self_collision.is_none()
            && self.cross_collision.is_none()
            && self.tear.is_none()
            && self.activation == 0.0
            && self.fibre.is_none()
    }

    /// The thermal rest-length multiplier `1 + α·(T − T₀)` (`1` when isothermal).
    pub fn thermal_stretch(&self) -> Real {
        match self.thermal {
            Some(t) => 1.0 + t.expansion * (t.temp - t.ambient),
            None => 1.0,
        }
    }

    /// The thermal stiffness multiplier `1 − coeff·(T − T₀)` (`1` when isothermal).
    pub fn thermal_softening(&self) -> Real {
        match self.thermal {
            Some(t) => 1.0 - t.stiffness_temp_coeff * (t.temp - t.ambient),
            None => 1.0,
        }
    }

    /// The effective rest-length multiplier: thermal expansion plus active strain.
    pub fn rest_scale(&self) -> Real {
        self.thermal_stretch() * (1.0 - self.activation.clamp(0.0, 1.0))
    }
}

/*
 * Accessors. Names are distinct from upstream's setters on purpose: upstream keeps
 * owning `set_material` / `set_volume_factor` / `tear_strain`, and everything here is
 * this fork's layer on top of it.
 */
impl crate::dynamics::soft_body::SoftBody {
    /// This fork's extension data for the body.
    pub fn fork_extras(&self) -> &ForkExtras {
        &self.fork
    }

    /// Mutable access (call [`Self::fork_touch`] afterwards if the change should wake
    /// the body and reach the solver this step).
    pub fn fork_extras_mut(&mut self) -> &mut ForkExtras {
        &mut self.fork
    }

    /// Marks the body modified so the next step re-reads the fork layer and wakes it.
    pub fn fork_touch(&mut self) {
        self.modified = true;
        self.wake_up();
    }

    /// Uniform wind / air-resistance field (`None` clears it).
    pub fn set_wind_field(&mut self, wind: Option<Wind>) {
        self.fork.wind = wind;
        self.fork_touch();
    }

    /// Uniform internal pressure `P` (force/area); `None` clears it.
    pub fn set_internal_pressure(&mut self, pressure: Option<Real>) {
        self.fork.pressure = pressure;
        self.fork_touch();
    }

    /// Body-wide internal damping in `[0, 1)`; out-of-range values are rejected.
    pub fn set_fork_damping(&mut self, damping: Real) -> bool {
        if !damping.is_finite() || !(0.0..1.0).contains(&damping) {
            return false;
        }
        self.fork.damping = damping;
        self.fork_touch();
        true
    }

    /// Uniform thermal field (`None` restores isothermal behaviour).
    pub fn set_thermal_field(&mut self, thermal: Option<ThermalParams>) -> bool {
        if let Some(t) = thermal
            && !(t.temp.is_finite()
                && t.ambient.is_finite()
                && t.expansion.is_finite()
                && t.stiffness_temp_coeff.is_finite())
        {
            return false;
        }
        self.fork.thermal = thermal;
        self.fork_touch();
        true
    }

    /// Rate-dependent stiffness layer (`None` restores pure elasticity).
    pub fn set_viscoelastic_layer(&mut self, viscoelastic: Option<ViscoelasticParams>) -> bool {
        if let Some(v) = viscoelastic
            && !(v.rate_coefficient.is_finite() && v.rate_coefficient >= 0.0)
        {
            return false;
        }
        self.fork.viscoelastic = viscoelastic;
        self.fork_touch();
        true
    }

    /// Body-level orthotropic stiffness axes (`None` makes every edge isotropic).
    pub fn set_anisotropy_axes(&mut self, axes: Option<Vector>) -> bool {
        if let Some(a) = axes
            && !(a.x >= 0.0
                && a.y >= 0.0
                && a.z >= 0.0
                && a.x.is_finite()
                && a.y.is_finite()
                && a.z.is_finite())
        {
            return false;
        }
        self.fork.anisotropy = axes;
        self.fork_touch();
        true
    }

    /// Cohesion (breakable glue) toward other bodies (`None` clears it).
    pub fn set_cohesion_glue(&mut self, cohesion: Option<CohesionParams>) -> bool {
        if let Some(c) = cohesion
            && !(c.radius.is_finite()
                && c.stiffness.is_finite()
                && c.break_distance.is_finite()
                && c.radius > 0.0)
        {
            return false;
        }
        self.fork.cohesion = cohesion;
        self.fork_touch();
        true
    }

    /// Global active strain `γ ∈ [0, 1]` (muscle contraction); `0` is passive.
    pub fn set_active_strain(&mut self, gamma: Real) -> bool {
        if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) {
            return false;
        }
        self.fork.activation = gamma;
        self.fork_touch();
        true
    }

    /// Muscle-fibre axis: the active strain acts along it (`None` = isotropic).
    pub fn set_fibre_axis(&mut self, fibre: Option<Vector>) -> bool {
        if let Some(f) = fibre
            && f.length_squared() <= 0.0
        {
            return false;
        }
        self.fork.fibre = fibre;
        self.fork_touch();
        true
    }

    /// Extra substeps per step (`1` keeps the official adaptive substepping alone).
    pub fn set_extra_substeps(&mut self, substeps: u32) -> bool {
        if substeps == 0 {
            return false;
        }
        self.fork.substeps = substeps;
        self.fork_touch();
        true
    }

    /// Tearing criterion (`None` leaves tearing entirely to upstream's thresholds).
    pub fn set_tear_criterion(&mut self, criterion: Option<ForkTearCriterion>) {
        // Strain and stress thresholds have an exact upstream counterpart: the official tear
        // load is already normalized by them, so they are written into the material. The energy
        // criterion has no upstream equivalent and is evaluated by the solver itself.
        match criterion {
            Some(ForkTearCriterion::Strain(threshold)) => {
                self.material.tear_strain = Some(threshold);
                self.material.tear_force = None;
            }
            Some(ForkTearCriterion::Stress(threshold)) => {
                self.material.tear_force = Some(threshold);
                self.material.tear_strain = None;
            }
            // The energy criterion replaces upstream's thresholds: it is evaluated by the solver,
            // so a stale `tear_strain`/`tear_force` would also tear on the old rule.
            Some(ForkTearCriterion::Energy(_)) => {
                self.material.tear_strain = None;
                self.material.tear_force = None;
            }
            None => {}
        }
        self.fork.tear = criterion;
        self.fork_touch();
    }

    /// Self-contact parameters (`None` disables this fork's self-contact layer).
    pub fn set_self_contact(&mut self, params: Option<SelfCollisionParams>) -> bool {
        if let Some(p) = params
            && !(p.radius.is_finite()
                && p.radius > 0.0
                && p.stiffness.is_finite()
                && (0.0..=1.0).contains(&p.friction))
        {
            return false;
        }
        self.fork.self_collision = params;
        self.fork_touch();
        true
    }

    /// Cross-body (soft-soft) contact parameters (`None` disables it).
    pub fn set_cross_contact(&mut self, params: Option<SelfCollisionParams>) -> bool {
        if let Some(p) = params
            && !(p.radius.is_finite()
                && p.radius > 0.0
                && p.stiffness.is_finite()
                && (0.0..=1.0).contains(&p.friction))
        {
            return false;
        }
        self.fork.cross_collision = params;
        self.fork_touch();
        true
    }
}
