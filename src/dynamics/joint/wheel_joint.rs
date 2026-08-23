//! A wheel joint is only meaningful in 3D (it uses yaw/roll axes that don't exist in 2D).

#![cfg(feature = "dim3")]

use crate::dynamics::JointAxis;
use crate::dynamics::joint::{GenericJoint, GenericJointBuilder, JointAxesMask, MotorModel};
use crate::math::{Real, Vector};

/// A wheel joint connecting two rigid-bodies, used for vehicle suspension and steering.
///
/// A wheel joint models a wheel attached to a chassis (or, more generally, two bodies
/// connected by a spring-loaded axle). It is composed of three conceptual behaviors, all
/// mapped onto the generic joint's motor/lock axes:
///
/// - **Suspension**: the `LIN_Y` axis acts as a spring-damper pulling the two bodies toward
///   a rest length along the joint's local Y axis (the suspension travel direction). Set it
///   with [`Self::set_suspension`]. This is what makes the wheel compress and rebound.
/// - **Steering**: the `ANG_Y` axis (yaw) controls the wheel's heading. Drive it with
///   [`Self::set_steering`] (a target angle motor). When no steering is set, the wheel is
///   free to yaw.
/// - **Axle drive**: the `ANG_X` axis (roll) drives the wheel's spin. Motorize it with
///   [`Self::set_axle_velocity`] / [`Self::set_axle_target`].
///
/// Tire/road friction is handled by the contact constraint between the wheel's collider and
/// the ground — the joint only constrains the suspension, steering, and spin. All linear
/// degrees of freedom other than the suspension axis, and all angular degrees of freedom
/// other than steering/spin, are locked.
///
/// This is a thin wrapper over [`GenericJoint`]; it reuses the existing constraint solver and
/// needs no engine-core changes.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct WheelJoint {
    /// The underlying joint data.
    pub data: GenericJoint,
}

impl WheelJoint {
    /// Creates a new wheel joint with the given suspension rest length, stiffness, and damping.
    ///
    /// The suspension travels along the joint's local Y axis; `rest_length` is its natural
    /// length, `stiffness`/`damping` are the spring and damper coefficients (force-based by
    /// default). The steering (`ANG_Y`) and spin (`ANG_X`) axes are left free until driven.
    pub fn new(rest_length: Real, stiffness: Real, damping: Real) -> Self {
        let data = GenericJointBuilder::new(
            // Lock everything except the suspension travel (LIN_Y), steering (ANG_Y) and spin (ANG_X).
            JointAxesMask::LIN_X | JointAxesMask::LIN_Z | JointAxesMask::ANG_Z,
        )
        .coupled_axes(JointAxesMask::LIN_Y)
        .motor_position(JointAxis::LinY, rest_length, stiffness, damping)
        .motor_model(JointAxis::LinY, MotorModel::ForceBased)
        .build();
        Self { data }
    }

    /// The underlying generic joint.
    pub fn data(&self) -> &GenericJoint {
        &self.data
    }

    /// Are contacts between the attached rigid-bodies enabled?
    pub fn contacts_enabled(&self) -> bool {
        self.data.contacts_enabled
    }

    /// Sets whether contacts between the attached rigid-bodies are enabled.
    pub fn set_contacts_enabled(&mut self, enabled: bool) -> &mut Self {
        self.data.set_contacts_enabled(enabled);
        self
    }

    /// The joint's anchor, expressed in the local-space of the first rigid-body.
    #[must_use]
    pub fn local_anchor1(&self) -> Vector {
        self.data.local_anchor1()
    }

    /// Sets the joint's anchor, expressed in the local-space of the first rigid-body.
    pub fn set_local_anchor1(&mut self, anchor1: Vector) -> &mut Self {
        self.data.set_local_anchor1(anchor1);
        self
    }

    /// The joint's anchor, expressed in the local-space of the second rigid-body.
    #[must_use]
    pub fn local_anchor2(&self) -> Vector {
        self.data.local_anchor2()
    }

    /// Sets the joint's anchor, expressed in the local-space of the second rigid-body.
    pub fn set_local_anchor2(&mut self, anchor2: Vector) -> &mut Self {
        self.data.set_local_anchor2(anchor2);
        self
    }

    /// The suspension rest length (natural length of the spring along the local Y axis).
    #[must_use]
    pub fn suspension_rest_length(&self) -> Real {
        self.data
            .limits(JointAxis::LinY)
            .expect("suspension axis is always limited")
            .max
    }

    /// Sets the suspension rest length.
    pub fn set_suspension_rest_length(&mut self, rest_length: Real) -> &mut Self {
        self.data.set_limits(JointAxis::LinY, [0.0, rest_length]);
        self
    }

    /// Sets the suspension spring stiffness and damping (force-based model), keeping the
    /// current rest length.
    pub fn set_suspension(&mut self, stiffness: Real, damping: Real) -> &mut Self {
        self.data.set_motor_position(
            JointAxis::LinY,
            self.suspension_rest_length(),
            stiffness,
            damping,
        );
        self.data
            .set_motor_model(JointAxis::LinY, MotorModel::ForceBased);
        self
    }

    /// Returns `(stiffness, damping)` of the suspension spring.
    #[must_use]
    pub fn suspension(&self) -> (Real, Real) {
        let m = self
            .data
            .motor(JointAxis::LinY)
            .expect("suspension axis is always motorized");
        (m.stiffness, m.damping)
    }

    /// Sets the steering target angle (in radians) for the yaw axis (ANG_Y), with the given
    /// stiffness and damping (an angular position motor). When not called, the wheel is free
    /// to yaw. Passing `target = 0.0` points the wheel straight ahead.
    pub fn set_steering(&mut self, target: Real, stiffness: Real, damping: Real) -> &mut Self {
        self.data
            .set_motor_position(JointAxis::AngY, target, stiffness, damping);
        self.data
            .set_motor_model(JointAxis::AngY, MotorModel::AccelerationBased);
        self
    }

    /// Sets the wheel's spin (ANG_X, roll) to a target velocity (rad/s), with the given
    /// damping (an angular velocity motor). Use for driven/braked wheels.
    pub fn set_axle_velocity(&mut self, target_vel: Real, damping: Real) -> &mut Self {
        self.data
            .set_motor_velocity(JointAxis::AngX, target_vel, damping);
        self.data
            .set_motor_model(JointAxis::AngX, MotorModel::AccelerationBased);
        self
    }

    /// Sets the wheel's spin (ANG_X, roll) to a target angle (radians), with the given
    /// stiffness and damping (an angular position motor).
    pub fn set_axle_target(&mut self, target: Real, stiffness: Real, damping: Real) -> &mut Self {
        self.data
            .set_motor_position(JointAxis::AngX, target, stiffness, damping);
        self.data
            .set_motor_model(JointAxis::AngX, MotorModel::AccelerationBased);
        self
    }
}

impl From<WheelJoint> for GenericJoint {
    fn from(val: WheelJoint) -> GenericJoint {
        val.data
    }
}

/// A [`WheelJoint`] builder using the builder pattern.
///
/// See the documentation of [`WheelJoint`] for the semantics of each setter.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct WheelJointBuilder(pub WheelJoint);

impl WheelJointBuilder {
    /// Creates a new builder for wheel joints.
    pub fn new(rest_length: Real, stiffness: Real, damping: Real) -> Self {
        Self(WheelJoint::new(rest_length, stiffness, damping))
    }

    /// Sets whether contacts between the attached rigid-bodies are enabled.
    #[must_use]
    pub fn contacts_enabled(mut self, enabled: bool) -> Self {
        self.0.set_contacts_enabled(enabled);
        self
    }

    /// Sets the joint's anchor, expressed in the local-space of the first rigid-body.
    #[must_use]
    pub fn local_anchor1(mut self, anchor1: Vector) -> Self {
        self.0.set_local_anchor1(anchor1);
        self
    }

    /// Sets the joint's anchor, expressed in the local-space of the second rigid-body.
    #[must_use]
    pub fn local_anchor2(mut self, anchor2: Vector) -> Self {
        self.0.set_local_anchor2(anchor2);
        self
    }

    /// Builds the wheel joint.
    pub fn build(self) -> WheelJoint {
        self.0
    }
}
