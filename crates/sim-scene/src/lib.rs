#[cfg(not(target_arch = "wasm32"))]
use anyhow::Context;
use anyhow::{Result, ensure};
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
#[cfg(not(target_arch = "wasm32"))]
use std::{fs, io::Write, path::Path};

pub const ONTOLOGY_VERSION: u32 = 1;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticClass {
    Ground,
    Gate,
    Obstacle,
}
impl SemanticClass {
    pub const fn id(self) -> u32 {
        match self {
            Self::Ground => 1,
            Self::Gate => 2,
            Self::Obstacle => 3,
        }
    }
}

pub const SCENE_VERSION: u32 = 2;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SensorConfig {
    pub eye: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub vertical_fov_radians: f32,
    pub near: f32,
    pub far: f32,
    pub width: u32,
    pub height: u32,
}

impl SensorConfig {
    /// Pixel coordinates have origin at the upper-left image edge; centers are (i+.5,j+.5).
    pub fn intrinsics(&self) -> [[f32; 3]; 3] {
        let f = self.height as f32 / (2.0 * (self.vertical_fov_radians * 0.5).tan());
        [
            [f, 0.0, self.width as f32 * 0.5],
            [0.0, f, self.height as f32 * 0.5],
            [0.0, 0.0, 1.0],
        ]
    }

    /// Optical coordinates: x right, y down, z forward. Returned matrix is column-major.
    pub fn world_to_optical(&self) -> [[f32; 4]; 4] {
        (Mat4::from_scale(Vec3::new(1.0, -1.0, -1.0))
            * Mat4::look_at_rh(self.eye.into(), self.target.into(), self.up.into()))
        .to_cols_array_2d()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneObject {
    pub id: u32,
    pub name: String,
    pub class: SemanticClass,
    pub parts: Vec<BoxPart>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoxPart {
    pub position: [f32; 3],
    pub scale: [f32; 3],
    pub color: [f32; 4],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneConfig {
    pub version: u32,
    pub sensor: SensorConfig,
    pub objects: Vec<SceneObject>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<Generation>,
}

impl Default for SceneConfig {
    fn default() -> Self {
        Self {
            version: SCENE_VERSION,
            sensor: SensorConfig {
                eye: [0.0, 2.0, 7.0],
                target: [0.0, 0.8, 0.0],
                up: [0.0, 1.0, 0.0],
                vertical_fov_radians: 55_f32.to_radians(),
                near: 0.1,
                far: 18.0,
                width: 640,
                height: 480,
            },
            generation: None,
            objects: vec![
                SceneObject {
                    id: 11,
                    name: "front red".into(),
                    class: SemanticClass::Obstacle,
                    parts: vec![BoxPart {
                        position: [-0.45, 0.7, 1.1],
                        scale: [1.4, 1.4, 1.4],
                        color: [0.7, 0.08, 0.04, 1.0],
                    }],
                },
                SceneObject {
                    id: 257,
                    name: "rear green".into(),
                    class: SemanticClass::Obstacle,
                    parts: vec![BoxPart {
                        position: [0.35, 0.9, -0.9],
                        scale: [1.8, 1.8, 1.8],
                        color: [0.08, 0.6, 0.13, 1.0],
                    }],
                },
                SceneObject {
                    id: 65539,
                    name: "right blue".into(),
                    class: SemanticClass::Obstacle,
                    parts: vec![BoxPart {
                        position: [2.1, 0.5, 0.1],
                        scale: [1.0, 1.0, 1.0],
                        color: [0.06, 0.18, 0.8, 1.0],
                    }],
                },
            ],
        }
    }
}

mod generator;
pub use generator::{GENERATOR_VERSION, Generation, GeneratorRecipe};

impl SceneConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == SCENE_VERSION,
            "unsupported scene version {}; expected {}",
            self.version,
            SCENE_VERSION
        );
        let s = &self.sensor;
        ensure!(
            (1..=2048).contains(&s.width) && (1..=2048).contains(&s.height),
            "sensor width and height must each be 1..=2048"
        );
        ensure!(
            s.eye
                .iter()
                .chain(&s.target)
                .chain(&s.up)
                .all(|v| v.is_finite() && v.abs() <= 1e6),
            "sensor vectors must be finite and within +/-1000000"
        );
        let forward = Vec3::from(s.target) - Vec3::from(s.eye);
        let up = Vec3::from(s.up);
        ensure!(forward.length() > 1e-5, "sensor eye and target must differ");
        ensure!(
            up.length() > 1e-5 && forward.normalize().cross(up.normalize()).length() > 1e-4,
            "sensor up must be nonzero and not parallel to viewing direction"
        );
        ensure!(
            s.vertical_fov_radians.is_finite() && (0.01..3.13).contains(&s.vertical_fov_radians),
            "sensor vertical_fov_radians must be in [0.01,3.13)"
        );
        ensure!(
            s.near.is_finite()
                && s.far.is_finite()
                && s.near >= 0.001
                && s.far > s.near
                && s.far <= 1e6,
            "sensor clipping planes require 0.001 <= near < far <= 1000000"
        );
        let view = Mat4::look_at_rh(s.eye.into(), s.target.into(), s.up.into());
        let projection = Mat4::perspective_rh(
            s.vertical_fov_radians,
            s.width as f32 / s.height as f32,
            s.near,
            s.far,
        );
        ensure!(
            view.is_finite() && (projection * view).is_finite(),
            "sensor camera produces a non-finite view/projection matrix"
        );
        if let Some(generation) = &self.generation {
            generation.recipe.validate()?;
        }
        ensure!(
            self.objects.len() <= 1024,
            "at most 1024 objects are supported"
        );
        let mut ids = HashSet::new();
        for o in &self.objects {
            ensure!(
                o.id != 0 && ids.insert(o.id),
                "object ID {} must be unique and nonzero (0 is background)",
                o.id
            );
            ensure!(
                !o.name.trim().is_empty()
                    && o.name.len() <= 128
                    && !o.name.chars().any(char::is_control),
                "object {} requires a nonempty name of at most 128 bytes without control characters",
                o.id
            );
            ensure!(
                !o.parts.is_empty() && o.parts.len() <= 64,
                "object {} requires 1..=64 box parts",
                o.id
            );
            for part in &o.parts {
                ensure!(
                    part.position
                        .iter()
                        .all(|v| v.is_finite() && v.abs() <= 1e6),
                    "object {} position must be finite and within +/-1000000",
                    o.id
                );
                ensure!(
                    part.scale
                        .iter()
                        .all(|v| v.is_finite() && *v >= 0.001 && *v <= 1e6),
                    "object {} scale must be finite in [0.001,1000000]",
                    o.id
                );
                ensure!(
                    part.color
                        .iter()
                        .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
                    "object {} color must be finite in [0,1]",
                    o.id
                );
            }
        }
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let config: Self = serde_json::from_reader(
            fs::File::open(path).with_context(|| format!("opening scene {}", path.display()))?,
        )
        .with_context(|| format!("decoding scene {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        atomic_write(path.as_ref(), &serde_json::to_vec_pretty(self)?)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    file.write_all(bytes)?;
    file.as_file().sync_all()?;
    file.persist(path)
        .with_context(|| format!("saving {}", path.display()))?;
    Ok(())
}
