//! Platform-independent trajectory protocol, timestamp playback and scene geometry.
use anyhow::{Context, Result, ensure};
use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};
use sim_graphics::{Camera, Frame, MeshData, MeshHandle, RenderPrimitive, Renderer, RendererError};

/// Scene version 1 reserves zero for sensor background, never visible geometry.
pub const GROUND_OBJECT_ID: u32 = u32::MAX;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraCalibration {
    pub vertical_fov_radians: f32,
    pub near: f32,
    pub far: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Header {
    pub version: u32,
    pub scene_version: u32,
    pub frame: String,
    pub quaternion: String,
    pub task: String,
    pub seed: u64,
    pub control_dt: f64,
    pub sample_every: u64,
    pub environment_ids: Vec<u32>,
    pub checkpoint: String,
    pub camera: CameraCalibration,
}

impl Header {
    pub fn parse(line: &str) -> Result<Self> {
        let header: Self = serde_json::from_str(line).context("invalid trajectory header")?;
        header.validate()?;
        Ok(header)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == 1 && self.scene_version == 1,
            "unsupported trajectory/scene version"
        );
        ensure!(
            self.frame == "ENU_FLU" && self.quaternion == "wxyz",
            "unsupported pose convention"
        );
        ensure!(
            matches!(
                self.task.as_str(),
                "hover" | "tracking" | "tracking-long-flight-v1"
            ),
            "unsupported task"
        );
        ensure!(
            self.control_dt.is_finite() && (self.control_dt - 0.01).abs() < 1e-12,
            "control_dt must be 0.01"
        );
        ensure!(self.sample_every > 0, "sample_every must be positive");
        ensure!(
            !self.environment_ids.is_empty() && self.environment_ids.len() <= 64,
            "select 1..64 environments"
        );
        for (index, id) in self.environment_ids.iter().enumerate() {
            ensure!(
                *id <= (u32::MAX - 2) / 2,
                "environment ID overflows scene object ID"
            );
            ensure!(
                !self.environment_ids[..index].contains(id),
                "duplicate environment ID"
            );
        }
        let c = &self.camera;
        ensure!(
            c.vertical_fov_radians.is_finite()
                && c.vertical_fov_radians > 0.0
                && c.vertical_fov_radians < std::f32::consts::PI,
            "invalid camera FOV"
        );
        ensure!(
            c.near.is_finite() && c.far.is_finite() && c.near > 0.0 && c.far > c.near,
            "invalid camera clipping planes"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VehiclePose {
    pub environment_id: u32,
    pub episode_id: u64,
    pub position_w: [f32; 3],
    pub attitude_wb: [f32; 4],
    pub target_w: [f32; 3],
}

impl VehiclePose {
    pub fn drone_object_id(&self) -> u32 {
        2 * self.environment_id + 1
    }

    pub fn target_object_id(&self) -> u32 {
        self.drone_object_id() + 1
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderSnapshot {
    pub step: u64,
    pub time_seconds: f64,
    pub vehicles: Vec<VehiclePose>,
}

impl RenderSnapshot {
    pub fn parse(line: &str, header: &Header, previous: Option<&Self>) -> Result<Self> {
        let snapshot: Self = serde_json::from_str(line).context("invalid trajectory snapshot")?;
        snapshot.validate(header, previous)?;
        Ok(snapshot)
    }

    pub fn validate(&self, header: &Header, previous: Option<&Self>) -> Result<()> {
        ensure!(
            self.time_seconds.is_finite() && self.time_seconds >= 0.0,
            "invalid snapshot timestamp"
        );
        let expected = self.step as f64 * header.control_dt;
        ensure!(
            (self.time_seconds - expected).abs() <= 1e-6_f64.max(expected.abs() * 1e-9),
            "timestamp does not match global step"
        );
        ensure!(
            self.vehicles.len() == header.environment_ids.len(),
            "incomplete environment selection"
        );
        if let Some(previous) = previous {
            ensure!(
                self.step > previous.step && self.time_seconds > previous.time_seconds,
                "snapshots must increase in global step/time"
            );
        }
        for (index, vehicle) in self.vehicles.iter().enumerate() {
            ensure!(
                header.environment_ids.contains(&vehicle.environment_id),
                "unselected environment"
            );
            ensure!(
                !self.vehicles[..index]
                    .iter()
                    .any(|v| v.environment_id == vehicle.environment_id),
                "duplicate snapshot environment"
            );
            ensure!(
                vehicle
                    .position_w
                    .iter()
                    .chain(vehicle.target_w.iter())
                    .chain(vehicle.attitude_wb.iter())
                    .all(|v| v.is_finite()),
                "nonfinite pose"
            );
            let norm: f32 = vehicle.attitude_wb.iter().map(|v| v * v).sum();
            ensure!(
                (norm - 1.0).abs() <= 1e-3,
                "attitude quaternion is not normalized"
            );
            if let Some(old) = previous.and_then(|s| {
                s.vehicles
                    .iter()
                    .find(|v| v.environment_id == vehicle.environment_id)
            }) {
                ensure!(vehicle.episode_id >= old.episode_id, "episode ID decreased");
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Recording {
    pub header: Header,
    pub snapshots: Vec<RenderSnapshot>,
}

impl Recording {
    pub fn parse(text: &str) -> Result<Self> {
        let mut lines = text.lines();
        let header = Header::parse(lines.next().context("empty recording")?)?;
        let mut snapshots = Vec::new();
        for (index, line) in lines.enumerate() {
            snapshots.push(
                RenderSnapshot::parse(line, &header, snapshots.last())
                    .with_context(|| format!("recording line {}", index + 2))?,
            );
        }
        ensure!(!snapshots.is_empty(), "recording has no snapshots");
        Ok(Self { header, snapshots })
    }

    /// Absolute simulation timestamp, sample-held through gaps and episode boundaries.
    pub fn sample(&self, time_seconds: f64) -> &RenderSnapshot {
        let index = self
            .snapshots
            .partition_point(|s| s.time_seconds <= time_seconds);
        &self.snapshots[index.saturating_sub(1)]
    }

    pub fn duration(&self) -> f64 {
        self.snapshots.last().unwrap().time_seconds - self.snapshots[0].time_seconds
    }
}

pub fn graphics_position(enu: [f32; 3]) -> Vec3 {
    Vec3::new(enu[0], enu[2], -enu[1])
}

fn attitude(vehicle: &VehiclePose) -> Quat {
    let [w, x, y, z] = vehicle.attitude_wb;
    // FLU to graphics is the same proper rotation as ENU to graphics.
    let basis = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
    basis * Quat::from_xyzw(x, y, z, w).normalize() * basis.conjugate()
}

#[derive(Clone, Copy, Debug)]
pub enum CameraMode {
    Orbit {
        azimuth: f32,
        elevation: f32,
        distance: f32,
    },
    Mounted {
        environment_id: u32,
    },
}

#[derive(Default)]
pub struct TrajectoryScene {
    cube: Option<MeshHandle>,
    plane: Option<MeshHandle>,
}

impl TrajectoryScene {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn initialize(&mut self, renderer: &mut Renderer) -> Result<(), RendererError> {
        self.cube = Some(renderer.register_mesh(MeshData::cube())?);
        self.plane = Some(renderer.register_mesh(MeshData::plane())?);
        Ok(())
    }

    pub fn camera(&self, header: &Header, snapshot: &RenderSnapshot, mode: CameraMode) -> Camera {
        let (eye, target, up) = match mode {
            CameraMode::Mounted { environment_id } => {
                let vehicle = snapshot
                    .vehicles
                    .iter()
                    .find(|v| v.environment_id == environment_id)
                    .unwrap_or(&snapshot.vehicles[0]);
                let rotation = attitude(vehicle);
                let eye =
                    graphics_position(vehicle.position_w) + rotation * Vec3::new(0.5, 0.12, 0.0);
                (eye, eye + rotation * Vec3::X, rotation * Vec3::Y)
            }
            CameraMode::Orbit {
                azimuth,
                elevation,
                distance,
            } => {
                let target = snapshot
                    .vehicles
                    .iter()
                    .map(|v| graphics_position(v.position_w))
                    .sum::<Vec3>()
                    / snapshot.vehicles.len() as f32;
                let elevation = elevation.clamp(0.05, 1.5);
                let offset = Vec3::new(
                    azimuth.cos() * elevation.cos(),
                    elevation.sin(),
                    azimuth.sin() * elevation.cos(),
                ) * distance.max(1.0);
                (target + offset, target, Vec3::Y)
            }
        };
        Camera {
            eye,
            target,
            up,
            vertical_fov_radians: header.camera.vertical_fov_radians,
            near: header.camera.near,
            far: header.camera.far,
        }
    }

    pub fn draw(&self, frame: &mut Frame, snapshot: &RenderSnapshot) {
        let (Some(cube), Some(plane)) = (self.cube, self.plane) else {
            return;
        };
        frame.draw(RenderPrimitive {
            mesh: plane,
            transform: Mat4::from_scale(Vec3::new(60.0, 1.0, 60.0)),
            color: [0.16, 0.20, 0.23, 1.0],
            object_id: GROUND_OBJECT_ID,
        });
        for vehicle in &snapshot.vehicles {
            let position = graphics_position(vehicle.position_w);
            let rotation = attitude(vehicle);
            let drone_id = vehicle.drone_object_id();
            let body = Mat4::from_rotation_translation(rotation, position);
            let mut part = |scale: Vec3, offset: Vec3, color: [f32; 4]| {
                frame.draw(RenderPrimitive {
                    mesh: cube,
                    transform: body
                        * Mat4::from_scale_rotation_translation(scale, Quat::IDENTITY, offset),
                    color,
                    object_id: drone_id,
                });
            };
            part(
                Vec3::new(0.55, 0.14, 0.24),
                Vec3::ZERO,
                [0.15, 0.55, 0.95, 1.0],
            );
            part(
                Vec3::new(0.16, 0.13, 0.16),
                Vec3::new(0.34, 0.0, 0.0),
                [1.0, 0.22, 0.06, 1.0],
            );
            for x in [-0.25, 0.25] {
                part(
                    Vec3::new(0.10, 0.07, 0.60),
                    Vec3::new(x, 0.0, 0.0),
                    [0.12, 0.14, 0.17, 1.0],
                );
                for z in [-0.3, 0.3] {
                    part(
                        Vec3::new(0.27, 0.025, 0.27),
                        Vec3::new(x, 0.10, z),
                        [0.75, 0.8, 0.84, 1.0],
                    );
                }
            }
            let target = graphics_position(vehicle.target_w);
            for scale in [
                Vec3::new(0.5, 0.045, 0.045),
                Vec3::new(0.045, 0.5, 0.045),
                Vec3::new(0.045, 0.045, 0.5),
            ] {
                frame.draw(RenderPrimitive {
                    mesh: cube,
                    transform: Mat4::from_scale_rotation_translation(scale, Quat::IDENTITY, target),
                    color: [0.2, 1.0, 0.35, 1.0],
                    object_id: vehicle.target_object_id(),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> Header {
        Header::parse(r#"{"version":1,"scene_version":1,"frame":"ENU_FLU","quaternion":"wxyz","task":"tracking","seed":1,"control_dt":0.01,"sample_every":5,"environment_ids":[7],"checkpoint":"policy.pt","camera":{"vertical_fov_radians":0.8,"near":0.1,"far":80.0}}"#).unwrap()
    }

    #[test]
    fn replay_holds_real_timestamps_across_drops_and_episode_changes() {
        let header = header();
        let snapshot = |step, episode_id, x| RenderSnapshot {
            step,
            time_seconds: step as f64 * 0.01,
            vehicles: vec![VehiclePose {
                environment_id: 7,
                episode_id,
                position_w: [x, 0.0, 1.0],
                attitude_wb: [1.0, 0.0, 0.0, 0.0],
                target_w: [0.0, 0.0, 1.0],
            }],
        };
        let frames = [
            snapshot(50, 0, 1.0),
            snapshot(100, 0, 2.0),
            snapshot(2000, 1, -1.0),
        ];
        let text = std::iter::once(serde_json::to_string(&header).unwrap())
            .chain(frames.iter().map(|s| serde_json::to_string(s).unwrap()))
            .collect::<Vec<_>>()
            .join("\n");
        let recording = Recording::parse(&text).unwrap();
        assert_eq!(recording.sample(0.75).vehicles[0].position_w[0], 1.0);
        assert_eq!(recording.sample(19.99).vehicles[0].position_w[0], 2.0);
        assert_eq!(recording.sample(20.0).vehicles[0].position_w[0], -1.0);
        assert_eq!(recording.sample(50.0).vehicles[0].episode_id, 1);
        let mut invalid = frames[2].clone();
        invalid.step = 100;
        invalid.time_seconds = 1.0;
        assert!(invalid.validate(&header, Some(&frames[1])).is_err());
    }

    #[test]
    fn mounted_camera_preserves_flu_forward_up_and_enu_yaw() {
        let mut snapshot = RenderSnapshot {
            step: 0,
            time_seconds: 0.0,
            vehicles: vec![VehiclePose {
                environment_id: 7,
                episode_id: 0,
                position_w: [2.0, 3.0, 4.0],
                attitude_wb: [1.0, 0.0, 0.0, 0.0],
                target_w: [0.0, 0.0, 1.0],
            }],
        };
        let scene = TrajectoryScene::new();
        let camera = scene.camera(
            &header(),
            &snapshot,
            CameraMode::Mounted { environment_id: 7 },
        );
        assert!(camera.eye.abs_diff_eq(Vec3::new(2.5, 4.12, -3.0), 1e-6));
        assert!((camera.target - camera.eye).abs_diff_eq(Vec3::X, 1e-6));
        assert!(camera.up.abs_diff_eq(Vec3::Y, 1e-6));
        let half = std::f32::consts::FRAC_1_SQRT_2;
        snapshot.vehicles[0].attitude_wb = [half, 0.0, 0.0, half];
        let camera = scene.camera(
            &header(),
            &snapshot,
            CameraMode::Mounted { environment_id: 7 },
        );
        assert!(camera.eye.abs_diff_eq(Vec3::new(2.0, 4.12, -3.5), 1e-6));
        assert!((camera.target - camera.eye).abs_diff_eq(-Vec3::Z, 1e-6));
        assert!(camera.up.abs_diff_eq(Vec3::Y, 1e-6));
    }
}
