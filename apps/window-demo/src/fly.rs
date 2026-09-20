//! Flyable drone through an infinite procedural world.
//!
//! Chunks load deterministically around the drone; each chunk is a pure
//! function of (seed, chunk_x, chunk_z). WASD + mouse to fly.
//! Run: `window-demo --fly [seed]`


use std::collections::{HashMap, HashSet};
use std::f32::consts::FRAC_PI_4;

use anyhow::Result;
use glam::{Mat4, Quat, Vec3};
use sim_graphics::{
    Camera, Frame, MeshData, MeshHandle, RenderPrimitive, RenderView, Renderer,
    RendererError, ViewKey, ViewKind, ViewOutputs,
};
use sim_graphics_winit::winit::event::{
    ElementState, MouseButton, MouseScrollDelta, WindowEvent,
};
use sim_graphics_winit::winit::keyboard::{KeyCode, PhysicalKey};
use sim_graphics_winit::Scene;

use crate::scene::{self, MeshKind, Prim};

const CHUNK: f32 = 24.0; // world units per chunk
const RADIUS: i32 = 2; // chunks loaded in each direction

fn mix64(mut v: u64) -> u64 {
    v ^= v >> 30;
    v = v.wrapping_mul(0xbf58476d1ce4e5b9);
    v ^= v >> 27;
    v = v.wrapping_mul(0x94d049bb133111eb);
    v ^ v >> 31
}

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
    // world-space noise: continuous across the whole map
    noise(x, z, 0.02) * 0.6 + noise(x, z, 0.08) * 0.3 + noise(x, z, 0.3) * 0.1
}

/// Primitives for one chunk, deterministic from (seed, cx, cz).
fn chunk(seed: u64, cx: i32, cz: i32) -> Vec<Prim> {
    let chunk_seed = mix64(
        seed ^ mix64(cx as u64).wrapping_mul(0x9e3779b9)
            ^ mix64(cz as u64).wrapping_mul(0x85ebca6b),
    );
    let origin = Vec3::new(cx as f32 * CHUNK, 0.0, cz as f32 * CHUNK);
    let mut prims = Vec::new();
    // terrain: flat grid displaced by world-space noise in the vertex shader
    prims.push(Prim {
        pos: origin,
        scale: Vec3::ONE,
        rot: Quat::IDENTITY,
        color: [0.25, 0.35, 0.20, 1.0],
        pattern: [9.0, 0.8, 0.0, 0.7], // kind 9 = terrain displace + noise texture
        mesh: MeshKind::Terrain,
        aux: [0.0; 4],
    });
    let scene_prims = scene::scene(chunk_seed);
    for mut p in scene_prims.0 {
        // scatter across the whole chunk, not just the corridor corner
        let jx = (mix64(chunk_seed ^ (p.pos.x.to_bits() as u64)) & 1023) as f32
            / 1023.0;
        let jz = (mix64(chunk_seed ^ (p.pos.z.to_bits() as u64) ^ 7) & 1023)
            as f32 / 1023.0;
        p.pos.x += origin.x + jx * CHUNK * 0.8 - CHUNK * 0.4;
        p.pos.z += origin.z + jz * CHUNK * 0.8 - CHUNK * 0.4;
        // snap at base: aux = (1, base_x, base_z, 0); pattern stays for texture
        p.aux = [1.0, p.pos.x, p.pos.z, 0.0];
        prims.push(p);
    }
    // extra dense fill: random objects across the chunk
    let n_fill = 20 + (mix64(chunk_seed ^ 99) & 1023) as usize % 15;
    for i in 0..n_fill {
        let b = (i * 5 + 200) as u64;
        let x = origin.x + (mix64(chunk_seed ^ b) & 1023) as f32 / 1023.0 * CHUNK;
        let z = origin.z
            + (mix64(chunk_seed ^ (b + 1)) & 1023) as f32 / 1023.0 * CHUNK;
        let h = 0.3 + (mix64(chunk_seed ^ (b + 2)) & 1023) as f32 / 1023.0 * 4.0;
        let w = 0.2 + (mix64(chunk_seed ^ (b + 3)) & 1023) as f32 / 1023.0 * 1.5;
        let is_cyl = (mix64(chunk_seed ^ (b + 4)) & 1023) as f32 / 1023.0 > 0.6;
        prims.push(Prim {
            pos: Vec3::new(x, h * 0.5, z),
            scale: Vec3::new(w, h, w * (0.5 + (mix64(chunk_seed ^ (b + 5)) & 1023) as f32 / 1023.0)),
            rot: Quat::IDENTITY,
            color: [
                0.1 + (mix64(chunk_seed ^ (b + 6)) & 1023) as f32 / 1023.0 * 0.7,
                0.1 + (mix64(chunk_seed ^ (b + 7)) & 1023) as f32 / 1023.0 * 0.7,
                0.1 + (mix64(chunk_seed ^ (b + 8)) & 1023) as f32 / 1023.0 * 0.7,
                1.0,
            ],
            pattern: [
                (mix64(chunk_seed ^ (b + 9)) & 1023) as f32 / 1023.0 * 9.0,
                0.5 + (mix64(chunk_seed ^ (b + 10)) & 1023) as f32 / 1023.0 * 3.0,
                (mix64(chunk_seed ^ (b + 11)) & 1023) as f32 / 1023.0 * 100.0,
                0.3 + (mix64(chunk_seed ^ (b + 12)) & 1023) as f32 / 1023.0 * 0.6,
            ],
            aux: [1.0, x, z, 0.0],
            mesh: if is_cyl { MeshKind::Cylinder } else { MeshKind::Cube },

        });
    }
    prims
}

struct Meshes {
    cube: MeshHandle,
    cylinder: MeshHandle,
    sphere: MeshHandle,
    terrain: MeshHandle,
    drone: MeshHandle,
}

pub struct FlyScene {
    frame: Frame,
    meshes: Option<Meshes>,
    seed: u64,
    // drone state
    pos: Vec3,
    vel: Vec3,
    yaw: f32,
    pitch: f32,
    // input
    keys: HashSet<KeyCode>,
    mouse_pressed: bool,
    last_mouse: Option<(f64, f64)>,
    // chunks
    loaded: HashMap<(i32, i32), Vec<Prim>>,
    last_chunk: (i32, i32),
    // camera mode
    fpv: bool,
    cam_dist: f32,
    cam_yaw: f32,
    cam_pitch: f32,
}

impl FlyScene {
    pub fn new(seed: u64) -> Self {
        Self {
            frame: Frame::new(),
            meshes: None,
            seed,
            pos: Vec3::new(0.0, 2.0, 0.0),
            vel: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            keys: HashSet::new(),
            mouse_pressed: false,
            last_mouse: None,
            loaded: HashMap::new(),
            last_chunk: (i32::MAX, i32::MAX),
            fpv: true,
            cam_dist: 4.0,
            cam_yaw: 0.0,
            cam_pitch: 0.4,
        }
    }

    fn chunk_coord(p: Vec3) -> (i32, i32) {
        (
            (p.x / CHUNK).floor() as i32,
            (p.z / CHUNK).floor() as i32,
        )
    }

    fn update_chunks(&mut self) {
        let (cx, cz) = Self::chunk_coord(self.pos);
        if (cx, cz) == self.last_chunk {
            return;
        }
        self.last_chunk = (cx, cz);
        // load missing
        for dx in -RADIUS..=RADIUS {
            for dz in -RADIUS..=RADIUS {
                let key = (cx + dx, cz + dz);
                if !self.loaded.contains_key(&key) {
                    let prims = chunk(self.seed, key.0, key.1);
                    self.loaded.insert(key, prims);
                }
            }
        }
        // unload far
        self.loaded.retain(|&(x, z), _| {
            (x - cx).abs() <= RADIUS && (z - cz).abs() <= RADIUS
        });
    }

    fn update_drone(&mut self, dt: f32) {
        // yaw/pitch from mouse
        let forward = Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            -self.yaw.cos() * self.pitch.cos(),
        );
        let right = Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin());
        let up = Vec3::Y;

        let mut thrust = Vec3::ZERO;
        if self.keys.contains(&KeyCode::KeyW) {
            thrust += forward;
        }
        if self.keys.contains(&KeyCode::KeyS) {
            thrust -= forward;
        }
        if self.keys.contains(&KeyCode::KeyA) {
            thrust -= right;
        }
        if self.keys.contains(&KeyCode::KeyD) {
            thrust += right;
        }
        if self.keys.contains(&KeyCode::Space) {
            thrust += up;
        }
        if self.keys.contains(&KeyCode::ShiftLeft) {
            thrust -= up;
        }

        let accel = 6.0;
        let drag = 3.0;
        self.vel += thrust * accel * dt;
        self.vel -= self.vel * drag * dt;
        self.pos += self.vel * dt;
        // soft floor
        if self.pos.y < 0.3 {
            self.pos.y = 0.3;
            self.vel.y = self.vel.y.max(0.0);
        }
    }
}

impl Scene for FlyScene {
    fn initialize(&mut self, renderer: &mut Renderer) -> Result<(), RendererError> {
        self.meshes = Some(Meshes {
            cube: renderer.register_mesh(MeshData::cube())?,
            cylinder: renderer.register_mesh(MeshData::cylinder(8))?,
            sphere: renderer.register_mesh(MeshData::sphere(8, 10))?,
            terrain: renderer.register_mesh(MeshData::grid(32, CHUNK))?,
            drone: renderer.register_mesh(MeshData::cube())?,
        });
        self.update_chunks();
        Ok(())
    }

    fn frame(&mut self, elapsed: f32, width: u32, height: u32) -> &Frame {
        let dt = 0.016f32.min(elapsed.max(0.001));
        self.update_drone(dt);
        self.update_chunks();
        let meshes = self.meshes.as_ref().expect("initialized");
        self.frame.begin();

        // camera
        let forward = Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            -self.yaw.cos() * self.pitch.cos(),
        );
        let (eye, target) = if self.fpv {
            (self.pos + Vec3::new(0.0, 0.15, 0.0), self.pos + forward * 5.0)
        } else {
            // third-person: orbit camera around the drone
            let eye = self.pos
                + Vec3::new(
                    self.cam_dist * self.cam_pitch.cos() * self.cam_yaw.sin(),
                    self.cam_dist * self.cam_pitch.sin(),
                    self.cam_dist * self.cam_pitch.cos() * self.cam_yaw.cos(),
                );
            (eye, self.pos)
        };
        self.frame.add_view(RenderView {
            key: ViewKey(0),
            kind: ViewKind::Display,
            camera: Camera {
                eye,
                target,
                up: Vec3::Y,
                vertical_fov_radians: FRAC_PI_4 * 1.1,
                near: 0.05,
                far: 300.0,
            },
            width,
            height,
            outputs: ViewOutputs::COLOR,
        });

        // draw drone (chase view only)
        if !self.fpv {
            self.frame.draw(RenderPrimitive {
                mesh: meshes.drone,
                transform: Mat4::from_scale_rotation_translation(
                    Vec3::new(0.3, 0.08, 0.3),
                    Quat::from_rotation_y(self.yaw),
                    self.pos,
                ),
                color: [0.9, 0.2, 0.2, 1.0],
                object_id: 9999,
                pattern: [0.0; 4],
                aux: [0.0; 4],
            });
        }



        // draw all loaded chunks
        for prims in self.loaded.values() {
            for p in prims {
                let mesh = match p.mesh {
                    MeshKind::Cube => meshes.cube,
                    MeshKind::Cylinder => meshes.cylinder,
                    MeshKind::Sphere => meshes.sphere,
                    MeshKind::Terrain => meshes.terrain,
                };
                self.frame.draw(RenderPrimitive {
                    mesh,
                    transform: Mat4::from_scale_rotation_translation(
                        p.scale, p.rot, p.pos,
                    ),
                    color: p.color,
                    object_id: 0,
                    pattern: p.pattern,
                    aux: p.aux,
                });
            }
        }
        &self.frame
    }

    fn window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            self.keys.insert(code);
                            if code == KeyCode::KeyV {
                                self.fpv = !self.fpv;
                            }
                        }
                        ElementState::Released => {
                            self.keys.remove(&code);
                        }
                    }
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.mouse_pressed = *state == ElementState::Pressed;
                if !self.mouse_pressed {
                    self.last_mouse = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    if let Some((lx, ly)) = self.last_mouse {
                        let dx = (position.x - lx) as f32;
                        let dy = (position.y - ly) as f32;
                        self.yaw += dx * 0.003;
                        self.pitch = (self.pitch - dy * 0.003).clamp(-1.2, 1.2);
                    }
                    self.last_mouse = Some((position.x, position.y));
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_x, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => (pos.y as f32) * 0.05,
                };
                self.cam_dist = (self.cam_dist + scroll * 0.5).clamp(1.0, 20.0);
            }
            _ => {}
        }
    }
}

pub fn run(seed: u64) -> Result<()> {
    sim_graphics_winit::run(
        format!("fly — seed {seed} | WASD+Space/Shift fly, drag look, V toggle FPV"),
        FlyScene::new(seed),
    )?;
    Ok(())
}
