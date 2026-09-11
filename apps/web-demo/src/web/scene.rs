use glam::{Mat4, Quat};
use sim_graphics::{MeshHandle, RenderPrimitive};
use sim_scene::SceneConfig;

/// GPU adapter only: generation and all realized geometry belong to sim-scene.
pub struct Scene {
    pub config: SceneConfig,
    pub primitives: Vec<RenderPrimitive>,
}

impl Scene {
    pub fn new(config: SceneConfig, cube: MeshHandle) -> Self {
        let primitives = config
            .objects
            .iter()
            .flat_map(|object| {
                object.parts.iter().map(move |part| RenderPrimitive {
                    mesh: cube,
                    object_id: object.id,
                    transform: Mat4::from_scale_rotation_translation(
                        part.scale.into(),
                        Quat::IDENTITY,
                        part.position.into(),
                    ),
                    color: part.color,
                })
            })
            .collect();
        Self { config, primitives }
    }
}

pub fn palette(id: u32) -> String {
    if id == 0 {
        return "#000000".into();
    }
    let mut hash = id;
    hash = (hash ^ (hash >> 16)).wrapping_mul(0x7feb352d);
    hash = (hash ^ (hash >> 15)).wrapping_mul(0x846ca68b);
    hash ^= hash >> 16;
    let channel = |shift: u32| 64 + ((hash >> shift) & 255u32) * 191 / 255;
    format!("#{:02x}{:02x}{:02x}", channel(0), channel(8), channel(16))
}
