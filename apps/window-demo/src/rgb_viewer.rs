//! Interactive viewer for the rgb-env corridor scene.
//!
//! Replicates the scene generation from apps/rgb-env (obstacles, clutter,
//! patterns, camera jitter, terrain, scenery) so the randomized environments
//! can be inspected with an orbit camera. Run: `window-demo --rgb [seed] [envs]`


use std::f32::consts::FRAC_PI_4;

use anyhow::Result;
use glam::{Mat4, Vec3};
use sim_graphics::{
    Camera, Frame, MeshData, MeshHandle, RenderPrimitive, RenderView, Renderer,
    RendererError, ViewKey, ViewKind, ViewOutputs,
};
use sim_graphics_winit::winit::event::{
    ElementState, MouseButton, MouseScrollDelta, WindowEvent,
};
use sim_graphics_winit::Scene;


// --- scene generation mirrored from apps/rgb-env/src/lib.rs ---
fn terrain_noise(x: f32, z: f32) -> f32 {
    let h = |ix: i32, iz: i32| {
        let v = mix64(
            (ix as u64).wrapping_mul(0x9e3779b97f4a7c15)
                ^ (iz as u64).wrapping_mul(0x85ebca6b),
        );
        (v & 1023) as f32 / 1023.0
    };
    let noise = |x: f32, z: f32, freq: f32| {
        let fx = x * freq;
        let fz = z * freq;
        let ix = fx.floor() as i32;
        let iz = fz.floor() as i32;
        let tx = fx - ix as f32;
        let tz = fz - iz as f32;
        let sx = tx * tx * (3.0 - 2.0 * tx);
        let sz = tz * tz * (3.0 - 2.0 * tz);
        let a = h(ix, iz);
        let b = h(ix + 1, iz);
        let c = h(ix, iz + 1);
        let d = h(ix + 1, iz + 1);
        a + (b - a) * sx + (c - a) * sz + (a - b - c + d) * sx * sz
    };
    noise(x, z, 0.35) * 0.7 + noise(x, z, 0.9) * 0.3
}

fn scenery(seed: u64, env: usize, episode: u64) -> Scenery {
    let stream = mix64(
        seed ^ mix64((env as u64).wrapping_mul(0x9e3779b97f4a7c15))
            ^ mix64(episode.wrapping_mul(0x85ebca6b))
            ^ 0x5ce9e9u64,
    );
    let j = |index: u64| {
        let v = mix64(stream.wrapping_add(index.wrapping_mul(0xc2b2ae35)));
        (v & 1023) as f32 / 1023.0
    };
    let mut buildings = [(Vec3::ZERO, Vec3::ZERO); 4];
    for i in 0..4 {
        let b = (i * 4) as u64;
        let side = if j(b) > 0.5 { 1.0 } else { -1.0 };
        let x = side * (5.5 + j(b + 1) * 3.0);
        let z = -1.0 - j(b + 2) * 8.0;
        let h = 1.5 + j(b + 3) * 3.5;
        let w = 0.8 + j(b + 1) * 1.4;
        buildings[i] = (
            Vec3::new(x, h * 0.5, z),
            Vec3::new(w, h, w * (0.6 + j(b + 2))),
        );
    }
    let mut trees = [(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO); 6];
    for i in 0..6 {
        let b = (i * 4 + 16) as u64;
        let side = if j(b) > 0.5 { 1.0 } else { -1.0 };
        let x = side * (3.0 + j(b + 1) * 5.0);
        let z = -0.5 - j(b + 2) * 8.5;
        let trunk_h = 0.4 + j(b + 3) * 0.8;
        let canopy_r = 0.3 + j(b + 1) * 0.5;
        trees[i] = (
            Vec3::new(x, trunk_h * 0.5, z),
            Vec3::new(0.08, trunk_h, canopy_r),
            Vec3::new(x, trunk_h + canopy_r * 0.7, z),
        );
    }
    Scenery { buildings, trees }
}

struct Scenery {
    buildings: [(Vec3, Vec3); 4],
    trees: [(Vec3, Vec3, Vec3); 6],
}


fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d049bb133111eb);
    value ^ value >> 31
}

fn obstacles(seed: u64, env: usize, episode: u64) -> [(Vec3, Vec3); 2] {
    let stream = mix64(
        seed ^ mix64((env as u64).wrapping_mul(0x9e3779b97f4a7c15))
            ^ mix64(episode.wrapping_mul(0x85ebca6b)),
    );
    let jitter = |index: u64, shift: u32| {
        let value = mix64(stream.wrapping_add(index.wrapping_mul(0xc2b2ae35)));
        ((value >> shift) & 1023) as f32 / 1023.0
    };
    let mut layout = [(Vec3::ZERO, Vec3::ZERO); 2];
    let gap_x = (jitter(4, 0) - 0.5) * 5.2;
    let gap_half = 1.5f32;
    let wall_z = -5.2f32;
    let wall_half_z = 0.15f32;
    let wall_y = 1.2f32;
    let corridor_half = 4.5f32;
    let left_width = (gap_x - gap_half) - (-corridor_half);
    let right_width = corridor_half - (gap_x + gap_half);
    layout[0] = (
        Vec3::new(-corridor_half + left_width * 0.5, wall_y * 0.5, wall_z),
        Vec3::new(left_width.max(0.01), wall_y, wall_half_z * 2.0),
    );
    layout[1] = (
        Vec3::new(gap_x + gap_half + right_width * 0.5, wall_y * 0.5, wall_z),
        Vec3::new(right_width.max(0.01), wall_y, wall_half_z * 2.0),
    );
    layout
}

fn clutter(seed: u64, env: usize, episode: u64) -> [(Vec3, Vec3); 6] {
    let stream = mix64(
        seed ^ mix64((env as u64).wrapping_mul(0x9e3779b97f4a7c15))
            ^ mix64(episode.wrapping_mul(0x85ebca6b))
            ^ 0xc1773du64,
    );
    let j = |index: u64| {
        let v = mix64(stream.wrapping_add(index.wrapping_mul(0xc2b2ae35)));
        (v & 1023) as f32 / 1023.0
    };
    let mut out = [(Vec3::ZERO, Vec3::ZERO); 6];
    for i in 0..6 {
        let base = (i * 4) as u64;
        let side = if j(base) > 0.5 { 1.0 } else { -1.0 };
        let x = side * (2.6 + j(base + 1) * 1.6);
        let z = -1.0 - j(base + 2) * 7.0;
        let h = 0.15 + j(base + 3) * 0.5;
        let w = 0.2 + j(base + 1) * 0.6;
        out[i] = (
            Vec3::new(x, h * 0.5, z),
            Vec3::new(w, h, w * (0.5 + j(base + 2))),
        );
    }
    out
}

fn surface_pattern(seed: u64, env: usize, index: usize, variant: u8) -> [f32; 4] {
    let stream = mix64(
        seed ^ mix64((env as u64).wrapping_mul(747796405))
            .wrapping_add((index as u64 + 1).wrapping_mul(2891336453))
            ^ mix64((variant as u64).wrapping_mul(0x27d4eb2f))
            ^ 0x9a77e9u64,
    );
    let j = |shift: u32| ((stream >> shift) & 1023) as f32 / 1023.0;
    let kind = if j(0) < 0.4 { 0.0 } else { (j(5) * 4.0).floor() + 1.0 };
    [kind, 0.5 + j(10) * 3.0, j(20) * 100.0, 0.15 + j(30) * 0.5]
}

fn floor_color(seed: u64, env: usize, variant: u8) -> [f32; 4] {
    let value = mix64(
        seed ^ mix64(env as u64).wrapping_add(0x9e3779b97f4a7c15)
            ^ mix64((variant as u64).wrapping_mul(0x27d4eb2f)),
    );
    [
        0.06 + (value & 1023) as f32 / 1023.0 * 0.55,
        0.06 + ((value >> 10) & 1023) as f32 / 1023.0 * 0.55,
        0.06 + ((value >> 20) & 1023) as f32 / 1023.0 * 0.55,
        1.0,
    ]
}

fn obstacle_color(seed: u64, env: usize, index: usize, variant: u8) -> [f32; 4] {
    let value = mix64(
        seed ^ mix64((env as u64).wrapping_mul(747796405))
            .wrapping_add((index as u64 + 1).wrapping_mul(2891336453))
            ^ mix64((variant as u64).wrapping_mul(0x27d4eb2f)),
    );
    [
        0.08 + (value & 1023) as f32 / 1023.0 * 0.80,
        0.08 + ((value >> 10) & 1023) as f32 / 1023.0 * 0.80,
        0.08 + ((value >> 20) & 1023) as f32 / 1023.0 * 0.80,
        1.0,
    ]
}

// --- viewer scene ---

struct Meshes {
    cube: MeshHandle,
    plane: MeshHandle,
    terrain: MeshHandle,
    cylinder: MeshHandle,
    sphere: MeshHandle,
}

pub struct RgbEnvScene {
    frame: Frame,
    meshes: Option<Meshes>,
    seed: u64,
    envs: usize,
    spacing: f32,
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
    camera_yaw: f32,
    camera_pitch: f32,
    camera_distance: f32,
    camera_target: Vec3,
}

impl RgbEnvScene {
    pub fn new(seed: u64, envs: usize) -> Self {
        Self {
            frame: Frame::new(),
            meshes: None,
            seed,
            envs,
            spacing: 16.0,
            mouse_pressed: false,
            last_mouse_pos: None,
            camera_yaw: 0.6,
            camera_pitch: 0.35,
            camera_distance: 18.0,
            camera_target: Vec3::new(0.0, 0.5, -3.0),
        }
    }
}

impl Scene for RgbEnvScene {
    fn initialize(&mut self, renderer: &mut Renderer) -> Result<(), RendererError> {
        self.meshes = Some(Meshes {
            cube: renderer.register_mesh(MeshData::cube())?,
            plane: renderer.register_mesh(MeshData::plane())?,
            terrain: renderer.register_mesh(MeshData::heightfield(
                24, 14.0, 0.6, terrain_noise,
            ))?,
            cylinder: renderer.register_mesh(MeshData::cylinder(8))?,
            sphere: renderer.register_mesh(MeshData::sphere(8, 10))?,
        });
        Ok(())
    }

    fn frame(&mut self, _elapsed: f32, width: u32, height: u32) -> &Frame {
        let meshes = self.meshes.as_ref().expect("initialized");
        self.frame.begin();

        let eye = self.camera_target
            + Vec3::new(
                self.camera_distance * self.camera_pitch.cos() * self.camera_yaw.sin(),
                self.camera_distance * self.camera_pitch.sin(),
                self.camera_distance * self.camera_pitch.cos() * self.camera_yaw.cos(),
            );
        self.frame.add_view(RenderView {
            key: ViewKey(0),
            kind: ViewKind::Display,
            camera: Camera {
                eye,
                target: self.camera_target,
                up: Vec3::Y,
                vertical_fov_radians: FRAC_PI_4 * 1.05,
                near: 0.1,
                far: 200.0,
            },
            width,
            height,
            outputs: ViewOutputs::COLOR,
        });

        let grid = (self.envs as f32).sqrt().ceil() as usize;
        for env in 0..self.envs {
            let gx = env % grid;
            let gz = env / grid;
            let origin = Vec3::new(gx as f32 * self.spacing, 0.0, gz as f32 * self.spacing);
            let _episode = 0u64;
            let variant = 0u8;

            // procedural scene: wall + buildings + trees + wires + cover
            let scene_seed = mix64(
                self.seed ^ mix64(env as u64),
            );
            let (prims, _gap) = crate::scene::scene(scene_seed);
            for (index, p) in prims.iter().enumerate() {
                let mesh = match p.mesh {
                    crate::scene::MeshKind::Cube => meshes.cube,
                    crate::scene::MeshKind::Cylinder => meshes.cylinder,
                    crate::scene::MeshKind::Sphere => meshes.sphere,
                    crate::scene::MeshKind::Terrain => meshes.terrain,
                };
                self.frame.draw(RenderPrimitive {
                    mesh,
                    transform: Mat4::from_scale_rotation_translation(
                        p.scale,
                        p.rot,
                        origin + p.pos,
                    ),
                    color: p.color,
                    object_id: env as u32 * 10 + 2 + index as u32,
                    pattern: p.pattern,
                    aux: p.aux,
                });
            }
            // terrain heightfield under the corridor
            self.frame.draw(RenderPrimitive {
                mesh: meshes.terrain,
                transform: Mat4::from_translation(
                    origin + Vec3::new(0.0, -0.06, -3.0),
                ),
                color: floor_color(self.seed, env, variant),
                object_id: env as u32 * 10 + 9,
                pattern: surface_pattern(self.seed, env, 12, variant),
                    aux: [0.0; 4],
            });
        }
        &self.frame
    }

    fn window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.mouse_pressed = *state == ElementState::Pressed;
                if !self.mouse_pressed {
                    self.last_mouse_pos = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    if let Some((lx, ly)) = self.last_mouse_pos {
                        let dx = (position.x - lx) as f32;
                        let dy = (position.y - ly) as f32;
                        self.camera_yaw -= dx * 0.005;
                        self.camera_pitch =
                            (self.camera_pitch + dy * 0.005).clamp(0.05, 1.50);
                    }
                    self.last_mouse_pos = Some((position.x, position.y));
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_x, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => (pos.y as f32) * 0.05,
                };
                self.camera_distance =
                    (self.camera_distance + scroll * 1.5).clamp(3.0, 80.0);
            }
            _ => {}
        }
    }
}

pub fn run(seed: u64, envs: usize) -> Result<()> {
    sim_graphics_winit::run(
        format!("rgb-env scenes (seed {seed}, {envs} envs)"),
        RgbEnvScene::new(seed, envs),
    )?;
    Ok(())
}
