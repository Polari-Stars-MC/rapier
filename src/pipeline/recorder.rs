#![cfg(all(feature = "alloc", feature = "serde-serialize"))]

//! Phase D — record / replay of physics state for Box3D.
//!
//! `PhysicsWorld` already bundles every piece of simulation state (bodies,
//! colliders, joints, narrow-phase manifolds, islands, broad-phase tree, …) and
//! derives `serde::Serialize`/`Deserialize` under the `serde-serialize` feature.
//! That makes a faithful recorder trivial: snapshot the world each step, serialize
//! the whole recording, and on replay restore snapshots back into a live world.
//!
//! The skipped fields of `PhysicsWorld` (`physics_pipeline`, `ccd_solver`) are
//! reconstructed via `Default` on deserialize, which is exactly what a fresh
//! replay needs — no pipeline workspace is carried across the boundary.
//!
//! For bit-exact replay across machines, build with `enhanced-determinism`. The
//! recorder/player themselves do not require it; determinism is a property of the
//! underlying simulation, not of the snapshot format.

use crate::alloc_prelude::*;
use crate::pipeline::PhysicsWorld;
use serde::{Deserialize, Serialize};

/// A single recorded frame: a full snapshot of the [`PhysicsWorld`] at a step.
#[derive(Serialize, Deserialize)]
pub struct WorldFrame {
    /// The step index this frame was captured at (0-based).
    pub step: u64,
    /// The full world state at capture time.
    pub world: PhysicsWorld,
}

/// Records a stream of [`PhysicsWorld`] snapshots for later replay or regression capture.
///
/// Each [`capture`](Self::capture) deep-copies the live world (via a bincode
/// round-trip) so the recording is independent of later mutation. The whole
/// recording serializes to/from bytes with [`to_bytes`](Self::to_bytes) /
/// [`from_bytes`](Self::from_bytes), suitable for on-disk golden files or streaming.
#[derive(Serialize, Deserialize)]
pub struct WorldRecorder {
    frames: Vec<WorldFrame>,
    next_step: u64,
}

impl WorldRecorder {
    /// Creates an empty recorder starting at step 0.
    pub fn new() -> Self {
        Self {
            frames: Vec::new(),
            next_step: 0,
        }
    }

    /// Deep-copies the current world state into a new frame.
    ///
    /// Call this right after each `world.step()` (or before it, depending on
    /// whether you want pre- or post-step snapshots).
    pub fn capture(&mut self, world: &PhysicsWorld) {
        // Deep copy via a bincode round-trip: keeps the recording independent of
        // the live world and exercises the same (de)serialization path used for
        // on-disk persistence.
        let bytes = bincode::serialize(world).expect("serialize PhysicsWorld");
        let world: PhysicsWorld =
            bincode::deserialize(&bytes).expect("deserialize PhysicsWorld (clone)");
        self.frames.push(WorldFrame {
            step: self.next_step,
            world,
        });
        self.next_step += 1;
    }

    /// Number of captured frames.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether no frames have been captured yet.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Returns the captured frame at `index`, if any.
    pub fn frame(&self, index: usize) -> Option<&WorldFrame> {
        self.frames.get(index)
    }

    /// Serializes the whole recording to bytes (bincode).
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }

    /// Deserializes a recording previously produced by [`to_bytes`](Self::to_bytes).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(bytes)
    }
}

impl Default for WorldRecorder {
    fn default() -> Self {
        Self::new()
    }
}

/// Replays recorded [`WorldFrame`]s back into a live [`PhysicsWorld`].
///
/// Restoring a frame overwrites *all* world state with the snapshot — pipeline
/// workspaces are rebuilt via `Default`, exactly as on a fresh load.
pub struct WorldPlayer {
    frames: Vec<WorldFrame>,
    cursor: usize,
}

impl WorldPlayer {
    /// Builds a player from a recording, taking ownership of its frames.
    pub fn from_recording(rec: WorldRecorder) -> Self {
        Self {
            frames: rec.frames,
            cursor: 0,
        }
    }

    /// Builds a player from bytes previously produced by
    /// [`WorldRecorder::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::Error> {
        Ok(Self {
            frames: bincode::deserialize::<WorldRecorder>(bytes)?.frames,
            cursor: 0,
        })
    }

    /// Number of frames available to replay.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether there are no frames to replay.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Returns the frame at `index` without advancing the replay cursor.
    pub fn frame(&self, index: usize) -> Option<&WorldFrame> {
        self.frames.get(index)
    }

    /// Resets the replay cursor back to the first frame.
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Restores the next recorded frame into `world`, overwriting all state.
    ///
    /// Returns `false` once every frame has been restored.
    pub fn restore_next(&mut self, world: &mut PhysicsWorld) -> bool {
        if self.cursor >= self.frames.len() {
            return false;
        }
        // Reconstruct from the stored (serialized) frame through the same byte
        // path a real on-disk load would use — this is the replay fidelity check.
        let bytes = bincode::serialize(&self.frames[self.cursor]).expect("serialize frame");
        let frame: WorldFrame = bincode::deserialize(&bytes).expect("deserialize frame");
        *world = frame.world;
        self.cursor += 1;
        true
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::math::Vector;
    use crate::prelude::{ColliderBuilder, RigidBodyBuilder};

    /// Phase D: a recorded falling-box simulation must round-trip through bytes
    /// and replay back to a fresh world with identical body transforms at every
    /// frame — proving the snapshot format is lossless and replay-faithful.
    #[test]
    fn record_replay_reproduces_transforms() {
        // --- Record ---
        let mut world = PhysicsWorld::default();
        world.integration_parameters.dt = 1.0 / 60.0;

        let (ball, _) = world.insert(
            RigidBodyBuilder::dynamic().translation(Vector::new(0.0, 10.0, 0.0)),
            ColliderBuilder::cuboid(0.5, 0.5, 0.5),
        );

        let steps = 60;
        let mut recorded: Vec<Vector> = Vec::with_capacity(steps);
        let mut rec = WorldRecorder::new();
        for _ in 0..steps {
            world.step();
            rec.capture(&world);
            recorded
                .push(world.bodies.get(ball).unwrap().translation());
        }

        assert_eq!(rec.len(), steps);

        // --- Serialize / deserialize the recording ---
        let bytes = rec.to_bytes().expect("record -> bytes");
        let player = WorldPlayer::from_bytes(&bytes).expect("bytes -> player");
        assert_eq!(player.len(), steps);

        // --- Replay into a fresh world ---
        let mut replay = PhysicsWorld::default();
        replay.integration_parameters.dt = 1.0 / 60.0;
        let mut replayed: Vec<Vector> = Vec::with_capacity(steps);
        let mut player = WorldPlayer::from_bytes(&bytes).expect("bytes -> player");
        assert_eq!(player.len(), steps);
        while player.restore_next(&mut replay) {
            replayed.push(replay.bodies.get(ball).unwrap().translation());
        }

        assert_eq!(replayed.len(), steps);

        for (orig, rep) in recorded.iter().zip(replayed.iter()) {
            let d = (orig.x - rep.x).abs() + (orig.y - rep.y).abs() + (orig.z - rep.z).abs();
            assert!(d < 1e-9, "replay transform diverged at frame");
        }

        // And the final resting transform matches exactly (same serialized bytes).
        let last = replayed.last().unwrap();
        let d = (recorded[steps - 1].x - last.x).abs()
            + (recorded[steps - 1].y - last.y).abs()
            + (recorded[steps - 1].z - last.z).abs();
        assert!(d < 1e-9, "final replay transform diverged");
    }
}
