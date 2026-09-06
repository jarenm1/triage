use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

use glam::{Mat4, Quat, Vec3};
use sim_graphics::{
    Camera, Frame, MeshData, MeshHandle, RenderPrimitive, RenderView, Renderer, RendererError,
    ViewKey, ViewKind, ViewOutputs,
};
use sim_graphics_winit::{
    winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    Scene,
};

const DRONE_COUNT: usize = 100;
const DEBRIS_COUNT: usize = 6;
const ARENA_HALF_SIZE: f32 = 42.0;
const ARENA_HEIGHT: f32 = 26.0;
const COLLISION_RADIUS: f32 = 0.85;

#[cfg(not(target_arch = "wasm32"))]
mod inspection;
#[cfg(not(target_arch = "wasm32"))]
mod trajectory;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mode = args.next();
    if matches!(
        mode.as_deref().and_then(|s| s.to_str()),
        Some("--trajectory" | "--live")
    ) {
        let value = args.next().ok_or_else(|| {
            anyhow::anyhow!("usage: window-demo --trajectory FILE | --live HOST:PORT")
        })?;
        anyhow::ensure!(args.next().is_none(), "unexpected trajectory arguments");
        return trajectory::run(
            mode.as_deref() == Some(std::ffi::OsStr::new("--live")),
            value,
        );
    }
    if mode.as_deref() == Some(std::ffi::OsStr::new("--inspect")) {
        let path = args.next().map(std::path::PathBuf::from);
        anyhow::ensure!(args.next().is_none(), "usage: window-demo --inspect [scene.json]");
        return inspection::run(path);
    }
    Ok(sim_graphics_winit::run("Autonomous Drone Swarm Simulator", DroneDemoScene::new())?)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn run_wasm() {
    let _ = sim_graphics_winit::run("Autonomous Drone Swarm Simulator", DroneDemoScene::new());
}

#[cfg(target_arch = "wasm32")]
fn main() {}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1 << 24) as f32
    }

    fn range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }

    fn random_point_in_arena(&mut self) -> Vec3 {
        Vec3::new(
            self.range(-ARENA_HALF_SIZE * 0.85, ARENA_HALF_SIZE * 0.85),
            self.range(4.0, ARENA_HEIGHT * 0.85),
            self.range(-ARENA_HALF_SIZE * 0.85, ARENA_HALF_SIZE * 0.85),
        )
    }

    fn random_unit_vector(&mut self) -> Vec3 {
        let theta = self.range(0.0, TAU);
        let z = self.range(-1.0, 1.0);
        let r = (1.0 - z * z).max(0.0).sqrt();
        Vec3::new(r * theta.cos(), r * theta.sin(), z)
    }
}

#[derive(Clone, Copy)]
struct Tower {
    position: Vec3,
    radius: f32,
    height: f32,
    color: [f32; 4],
    beacon_color: [f32; 4],
    aim_dir: Vec3,
    fire_timer: f32,
}

#[derive(Clone, Copy)]
struct Projectile {
    position: Vec3,
    velocity: Vec3,
    lifetime: f32,
    color: [f32; 4],
}

#[derive(Clone, Copy)]
struct LaunchPad {
    position: Vec3,
    size: f32,
    color: [f32; 4],
}

#[derive(Clone, Copy)]
struct SquareGate {
    position: Vec3,
    size: f32,
    yaw: f32,
    color: [f32; 4],
    hit_timer: f32,
}

enum DroneState {
    Flying {
        target: Vec3,
        waypoint_timer: f32,
    },
    Crashing {
        elapsed: f32,
        duration: f32,
        debris_pos: [Vec3; DEBRIS_COUNT],
        debris_vel: [Vec3; DEBRIS_COUNT],
        debris_rot: [Quat; DEBRIS_COUNT],
        debris_ang_vel: [Vec3; DEBRIS_COUNT],
    },
    Respawning {
        timer: f32,
        pad_index: usize,
    },
    Takeoff {
        elapsed: f32,
        duration: f32,
        pad_pos: Vec3,
        target_alt: f32,
    },
}

struct Drone {
    id: u32,
    position: Vec3,
    velocity: Vec3,
    rotation: Quat,
    state: DroneState,
    color: [f32; 4],
    beacon_color: [f32; 4],
    max_speed: f32,
    max_accel: f32,
    propeller_angle: f32,
}

impl Drone {
    fn is_active(&self) -> bool {
        matches!(self.state, DroneState::Flying { .. } | DroneState::Takeoff { .. })
    }

    fn trigger_crash(&mut self, rng: &mut Rng) {
        let mut debris_pos = [self.position; DEBRIS_COUNT];
        let mut debris_vel = [Vec3::ZERO; DEBRIS_COUNT];
        let mut debris_rot = [self.rotation; DEBRIS_COUNT];
        let mut debris_ang_vel = [Vec3::ZERO; DEBRIS_COUNT];

        for i in 0..DEBRIS_COUNT {
            let offset = rng.random_unit_vector() * 0.3;
            debris_pos[i] = self.position + offset;
            let explode_dir = rng.random_unit_vector();
            let explode_speed = rng.range(4.0, 12.0);
            debris_vel[i] = explode_dir * explode_speed + self.velocity * 0.35 + Vec3::Y * 2.0;
            debris_rot[i] = Quat::from_scaled_axis(rng.random_unit_vector() * rng.range(0.0, PI));
            debris_ang_vel[i] = rng.random_unit_vector() * rng.range(6.0, 20.0);
        }

        self.state = DroneState::Crashing {
            elapsed: 0.0,
            duration: 1.35,
            debris_pos,
            debris_vel,
            debris_rot,
            debris_ang_vel,
        };
        self.velocity = Vec3::ZERO;
    }
}

struct SceneMeshes {
    cube: MeshHandle,
    plane: MeshHandle,
    cylinder: MeshHandle,
    sphere: MeshHandle,
}

pub struct DroneDemoScene {
    frame: Frame,
    meshes: Option<SceneMeshes>,
    drones: Vec<Drone>,
    towers: Vec<Tower>,
    pads: Vec<LaunchPad>,
    gates: Vec<SquareGate>,
    projectiles: Vec<Projectile>,
    rng: Rng,
    last_time: f32,
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
    camera_yaw: f32,
    camera_pitch: f32,
    camera_distance: f32,
    camera_target: Vec3,
    is_user_interacting: bool,
}

const TEAM_COLORS: [([f32; 4], [f32; 4]); 8] = [
    ([0.15, 0.78, 0.95, 1.0], [0.30, 0.95, 1.0, 1.0]), // Cyan
    ([0.98, 0.45, 0.12, 1.0], [1.0, 0.65, 0.2, 1.0]),  // Orange
    ([0.95, 0.18, 0.65, 1.0], [1.0, 0.35, 0.85, 1.0]), // Magenta
    ([0.22, 0.88, 0.45, 1.0], [0.45, 1.0, 0.60, 1.0]), // Emerald
    ([0.95, 0.85, 0.18, 1.0], [1.0, 0.95, 0.40, 1.0]), // Amber
    ([0.65, 0.35, 0.95, 1.0], [0.85, 0.55, 1.0, 1.0]), // Violet
    ([0.18, 0.45, 0.98, 1.0], [0.40, 0.70, 1.0, 1.0]), // Electric Blue
    ([0.95, 0.25, 0.25, 1.0], [1.0, 0.45, 0.45, 1.0]), // Crimson
];

impl DroneDemoScene {
    pub fn new() -> Self {
        let mut rng = Rng::new(1337);

        // 8 Launch Pads placed symmetrically in the arena
        let pads = vec![
            LaunchPad {
                position: Vec3::new(-28.0, 0.05, -28.0),
                size: 5.2,
                color: [0.05, 0.06, 0.08, 1.0],
            },
            LaunchPad {
                position: Vec3::new(28.0, 0.05, -28.0),
                size: 5.2,
                color: [0.05, 0.06, 0.08, 1.0],
            },
            LaunchPad {
                position: Vec3::new(-28.0, 0.05, 28.0),
                size: 5.2,
                color: [0.05, 0.06, 0.08, 1.0],
            },
            LaunchPad {
                position: Vec3::new(28.0, 0.05, 28.0),
                size: 5.2,
                color: [0.05, 0.06, 0.08, 1.0],
            },
            LaunchPad {
                position: Vec3::new(0.0, 0.05, -34.0),
                size: 5.2,
                color: [0.05, 0.06, 0.08, 1.0],
            },
            LaunchPad {
                position: Vec3::new(0.0, 0.05, 34.0),
                size: 5.2,
                color: [0.05, 0.06, 0.08, 1.0],
            },
            LaunchPad {
                position: Vec3::new(-34.0, 0.05, 0.0),
                size: 5.2,
                color: [0.05, 0.06, 0.08, 1.0],
            },
            LaunchPad {
                position: Vec3::new(34.0, 0.05, 0.0),
                size: 5.2,
                color: [0.05, 0.06, 0.08, 1.0],
            },
        ];

        // 2 Short Defense Turret Bunkers (Low Profile)
        let pillar_color = [0.08, 0.09, 0.12, 1.0];
        let towers = vec![
            Tower {
                position: Vec3::new(-28.0, 0.0, -18.0),
                radius: 2.6,
                height: 8.5,
                color: pillar_color,
                beacon_color: [1.0, 0.25, 0.25, 1.0], // Crimson plasma turret
                aim_dir: Vec3::new(1.0, 0.3, 0.5).normalize(),
                fire_timer: 0.05,
            },
            Tower {
                position: Vec3::new(28.0, 0.0, 18.0),
                radius: 2.6,
                height: 8.5,
                color: pillar_color,
                beacon_color: [0.15, 0.90, 1.0, 1.0], // Cyan plasma turret
                aim_dir: Vec3::new(-1.0, 0.3, -0.5).normalize(),
                fire_timer: 0.10,
            },
        ];

        // 6 Sky Waypoint Square Gates (Elevated Airspace, 100% Clear of Towers)
        let default_gate_color = [0.45, 0.47, 0.52, 1.0];
        let gates = vec![
            SquareGate {
                position: Vec3::new(0.0, 14.0, -22.0),
                size: 3.8,
                yaw: FRAC_PI_2,
                color: default_gate_color,
                hit_timer: 0.0,
            },
            SquareGate {
                position: Vec3::new(0.0, 14.0, 22.0),
                size: 3.8,
                yaw: FRAC_PI_2,
                color: default_gate_color,
                hit_timer: 0.0,
            },
            SquareGate {
                position: Vec3::new(22.0, 15.0, -8.0),
                size: 3.8,
                yaw: 0.0,
                color: default_gate_color,
                hit_timer: 0.0,
            },
            SquareGate {
                position: Vec3::new(-22.0, 15.0, 8.0),
                size: 3.8,
                yaw: 0.0,
                color: default_gate_color,
                hit_timer: 0.0,
            },
            SquareGate {
                position: Vec3::new(0.0, 20.0, 0.0),
                size: 4.2,
                yaw: 0.0,
                color: default_gate_color,
                hit_timer: 0.0,
            },
            SquareGate {
                position: Vec3::new(-12.0, 16.0, -12.0),
                size: 3.8,
                yaw: FRAC_PI_4,
                color: default_gate_color,
                hit_timer: 0.0,
            },
        ];

        // 100 Initial Drones
        let mut drones = Vec::with_capacity(DRONE_COUNT);
        for i in 0..DRONE_COUNT {
            let (color, beacon_color) = TEAM_COLORS[i % TEAM_COLORS.len()];
            let pad_idx = i % pads.len();
            let pad_pos = pads[pad_idx].position;
            let offset_x = (rng.next_f32() - 0.5) * 3.5;
            let offset_z = (rng.next_f32() - 0.5) * 3.5;
            let start_pos = Vec3::new(pad_pos.x + offset_x, 0.4, pad_pos.z + offset_z);

            // Stagger drone initial states
            let state = if i < 60 {
                let fly_target = rng.random_point_in_arena();
                DroneState::Flying {
                    target: fly_target,
                    waypoint_timer: rng.range(0.0, 5.0),
                }
            } else if i < 85 {
                let target_alt = rng.range(6.0, 20.0);
                DroneState::Takeoff {
                    elapsed: rng.range(0.0, 2.0),
                    duration: 2.2,
                    pad_pos: start_pos,
                    target_alt,
                }
            } else {
                DroneState::Respawning {
                    timer: rng.range(0.2, 2.0),
                    pad_index: pad_idx,
                }
            };

            let pos = match state {
                DroneState::Flying { .. } => rng.random_point_in_arena(),
                _ => start_pos,
            };

            drones.push(Drone {
                id: (i as u32) + 10,
                position: pos,
                velocity: rng.random_unit_vector() * rng.range(2.0, 10.0),
                rotation: Quat::IDENTITY,
                state,
                color,
                beacon_color,
                max_speed: rng.range(14.0, 24.0),
                max_accel: rng.range(22.0, 36.0),
                propeller_angle: rng.range(0.0, TAU),
            });
        }

        Self {
            frame: Frame::with_capacity(3500, 2),
            meshes: None,
            drones,
            towers,
            pads,
            gates,
            projectiles: Vec::with_capacity(64),
            rng,
            last_time: 0.0,
            mouse_pressed: false,
            last_mouse_pos: None,
            camera_yaw: 0.8,
            camera_pitch: 0.48,
            camera_distance: 52.0,
            camera_target: Vec3::new(0.0, 7.0, 0.0),
            is_user_interacting: false,
        }
    }
    fn update_simulation(&mut self, dt: f32) {
        let drone_count = self.drones.len();
        let positions: Vec<Vec3> = self.drones.iter().map(|d| d.position).collect();

        // Decay gate hit flash timers
        for gate in &mut self.gates {
            gate.hit_timer = (gate.hit_timer - dt * 2.0).max(0.0);
        }

        // Spatial Grid for O(N) Flocking & Collision Detection
        const GRID_X: usize = 22;
        const GRID_Y: usize = 7;
        const GRID_Z: usize = 22;
        const CELL_COUNT: usize = GRID_X * GRID_Y * GRID_Z;

        let mut grid_head = [u32::MAX; CELL_COUNT];
        let mut grid_next = vec![u32::MAX; drone_count];

        for (idx, &pos) in positions.iter().enumerate() {
            if self.drones[idx].is_active() {
                let gx = ((pos.x + 44.0) * 0.25).clamp(0.0, (GRID_X - 1) as f32) as usize;
                let gy = (pos.y * 0.25).clamp(0.0, (GRID_Y - 1) as f32) as usize;
                let gz = ((pos.z + 44.0) * 0.25).clamp(0.0, (GRID_Z - 1) as f32) as usize;
                let cell = gx * (GRID_Y * GRID_Z) + gy * GRID_Z + gz;
                grid_next[idx] = grid_head[cell];
                grid_head[cell] = idx as u32;
            }
        }

        let mut crash_list = Vec::new();
        let mut new_projectiles = Vec::new();

        // Turrets Target Tracking and Gun Fire Logic (Slow Rotation & Elevation Lock)
        for tower in &mut self.towers {
            let turret_pos = tower.position + Vec3::new(0.0, tower.height + 0.8, 0.0);
            let mut best_target = None;
            let mut best_dist_sq = 80.0 * 80.0;

            for drone in &self.drones {
                if drone.is_active() && drone.position.y > turret_pos.y - 4.0 {
                    let diff = drone.position - turret_pos;
                    let d_sq = diff.length_squared();
                    if d_sq < best_dist_sq {
                        best_dist_sq = d_sq;
                        best_target = Some((diff, drone.velocity));
                    }
                }
            }

            if let Some((diff, _vel)) = best_target {
                let mut target_dir = diff.normalize_or_zero();
                // Lock: cannot aim below horizon (y >= 0.0)
                target_dir.y = target_dir.y.max(0.0);
                target_dir = target_dir.normalize_or_zero();

                // Slow, realistic turret rotation speed (~1.6 rad/s)
                let max_rot_speed = 1.6;
                let dot = tower.aim_dir.dot(target_dir).clamp(-1.0, 1.0);
                let angle = dot.acos();
                let max_step = max_rot_speed * dt;
                if angle > 1e-4 {
                    let step = (max_step / angle).min(1.0);
                    let mut new_aim = tower.aim_dir.lerp(target_dir, step);
                    new_aim.y = new_aim.y.max(0.0);
                    tower.aim_dir = new_aim.normalize_or_zero();
                }

                // Constantly shoot 10 rounds per second whenever targeting
                tower.fire_timer -= dt;
                if tower.fire_timer <= 0.0 {
                    tower.fire_timer = 0.10;
                    let proj_speed = 95.0;
                    let fire_dir = tower.aim_dir;

                    new_projectiles.push(Projectile {
                        position: turret_pos + fire_dir * 2.2,
                        velocity: fire_dir * proj_speed,
                        lifetime: 1.5,
                        color: tower.beacon_color,
                    });
                }
            } else {
                tower.fire_timer -= dt;
                if tower.aim_dir.y < 0.0 {
                    tower.aim_dir.y = 0.0;
                    tower.aim_dir = tower.aim_dir.normalize_or_zero();
                }
            }
        }
        self.projectiles.extend(new_projectiles);

        // Update Projectiles and Collide with Drones
        let mut p_idx = 0;
        while p_idx < self.projectiles.len() {
            let proj = &mut self.projectiles[p_idx];
            proj.position += proj.velocity * dt;
            proj.lifetime -= dt;

            let mut hit = false;
            if proj.lifetime > 0.0 {
                let p_pos = proj.position;
                let gx = ((p_pos.x + 44.0) * 0.25).clamp(0.0, (GRID_X - 1) as f32) as isize;
                let gy = (p_pos.y * 0.25).clamp(0.0, (GRID_Y - 1) as f32) as isize;
                let gz = ((p_pos.z + 44.0) * 0.25).clamp(0.0, (GRID_Z - 1) as f32) as isize;

                'proj_collision: for dx in -1..=1 {
                    let nx = gx + dx;
                    if nx < 0 || nx >= GRID_X as isize { continue; }
                    for dy in -1..=1 {
                        let ny = gy + dy;
                        if ny < 0 || ny >= GRID_Y as isize { continue; }
                        for dz in -1..=1 {
                            let nz = gz + dz;
                            if nz < 0 || nz >= GRID_Z as isize { continue; }
                            let n_cell = (nx as usize) * (GRID_Y * GRID_Z) + (ny as usize) * GRID_Z + (nz as usize);
                            let mut curr = grid_head[n_cell];
                            while curr != u32::MAX {
                                let d_i = curr as usize;
                                let drone = &mut self.drones[d_i];
                                if drone.is_active() {
                                    let diff = drone.position - p_pos;
                                    if diff.length_squared() < (COLLISION_RADIUS + 0.65) * (COLLISION_RADIUS + 0.65) {
                                        drone.trigger_crash(&mut self.rng);
                                        hit = true;
                                        break 'proj_collision;
                                    }
                                }
                                curr = grid_next[d_i];
                            }
                        }
                    }
                }
            }

            if hit || proj.lifetime <= 0.0 {
                self.projectiles.swap_remove(p_idx);
            } else {
                p_idx += 1;
            }
        }

        // 1. Update State Transitions & Navigation Steering
        for i in 0..drone_count {
            let mut new_state = None;
            let mut crash_now = false;
            let pad_count = self.pads.len();

            {
                let drone = &mut self.drones[i];
                drone.propeller_angle += (35.0 + drone.velocity.length() * 2.5) * dt;

                match &mut drone.state {
                    DroneState::Flying {
                        target,
                        waypoint_timer,
                    } => {
                        *waypoint_timer += dt;

                        // Target distance check or timeout
                        let to_target = *target - drone.position;
                        if to_target.length() < 3.0 || *waypoint_timer > 6.0 {
                            *target = self.rng.random_point_in_arena();
                            *waypoint_timer = 0.0;
                        }

                        // Target seeking force
                        let desired_dir = to_target.normalize_or_zero();
                        let desired_vel = desired_dir * drone.max_speed;
                        let mut steer = (desired_vel - drone.velocity) * 2.8;

                        // Obstacle Avoidance (Towers)
                        for tower in &self.towers {
                            if drone.position.y < tower.height + 2.0 {
                                let to_tower_xz = Vec3::new(
                                    drone.position.x - tower.position.x,
                                    0.0,
                                    drone.position.z - tower.position.z,
                                );
                                let dist_xz = to_tower_xz.length();
                                let min_dist = tower.radius + 3.2;
                                if dist_xz < min_dist && dist_xz > 0.01 {
                                    let push = to_tower_xz.normalize()
                                        * ((min_dist - dist_xz) / min_dist)
                                        * 45.0;
                                    steer += push;
                                    if drone.position.y > tower.height - 2.0 {
                                        steer.y += 18.0;
                                    }
                                }
                            }
                        }

                        // Boundary avoidance (contain within arena box)
                        if drone.position.x.abs() > ARENA_HALF_SIZE - 4.0 {
                            steer.x += -drone.position.x.signum() * 35.0;
                        }
                        if drone.position.z.abs() > ARENA_HALF_SIZE - 4.0 {
                            steer.z += -drone.position.z.signum() * 35.0;
                        }
                        if drone.position.y < 3.5 {
                            steer.y += (3.5 - drone.position.y) * 25.0;
                        } else if drone.position.y > ARENA_HEIGHT - 3.0 {
                            steer.y += (ARENA_HEIGHT - 3.0 - drone.position.y) * 25.0;
                        }

                        // Spatial Grid Flocking Separation & Collision Check
                        let gx = ((drone.position.x + 44.0) * 0.25).clamp(0.0, (GRID_X - 1) as f32) as isize;
                        let gy = (drone.position.y * 0.25).clamp(0.0, (GRID_Y - 1) as f32) as isize;
                        let gz = ((drone.position.z + 44.0) * 0.25).clamp(0.0, (GRID_Z - 1) as f32) as isize;

                        for dx in -1..=1 {
                            let nx = gx + dx;
                            if nx < 0 || nx >= GRID_X as isize { continue; }
                            for dy in -1..=1 {
                                let ny = gy + dy;
                                if ny < 0 || ny >= GRID_Y as isize { continue; }
                                for dz in -1..=1 {
                                    let nz = gz + dz;
                                    if nz < 0 || nz >= GRID_Z as isize { continue; }
                                    let n_cell = (nx as usize) * (GRID_Y * GRID_Z) + (ny as usize) * GRID_Z + (nz as usize);
                                    let mut curr = grid_head[n_cell];
                                    while curr != u32::MAX {
                                        let j = curr as usize;
                                        if i != j {
                                            let diff = drone.position - positions[j];
                                            let dist_sq = diff.length_squared();
                                            if dist_sq < 9.0 && dist_sq > 0.001 {
                                                let dist = dist_sq.sqrt();
                                                steer += (diff / dist) * ((3.0 - dist) / 3.0) * 22.0;
                                            }
                                            if dist_sq < COLLISION_RADIUS * COLLISION_RADIUS && j > i {
                                                crash_list.push(i);
                                                crash_list.push(j);
                                            }
                                        }
                                        curr = grid_next[j];
                                    }
                                }
                            }
                        }

                        // Clamp acceleration
                        let accel_mag = steer.length();
                        if accel_mag > drone.max_accel {
                            steer = (steer / accel_mag) * drone.max_accel;
                        }

                        // Integrate physics
                        drone.velocity += steer * dt;
                        let speed = drone.velocity.length();
                        if speed > drone.max_speed {
                            drone.velocity = (drone.velocity / speed) * drone.max_speed;
                        }
                        drone.position += drone.velocity * dt;

                        // Attitude calculation (yaw towards velocity + tilt into acceleration)
                        let horizontal_vel = Vec3::new(drone.velocity.x, 0.0, drone.velocity.z);
                        if horizontal_vel.length_squared() > 0.05 {
                            let yaw = -horizontal_vel.z.atan2(horizontal_vel.x) + FRAC_PI_2;
                            let forward_tilt = (drone.velocity.y * 0.05).clamp(-0.4, 0.4);
                            let bank_tilt = -(steer.dot(Vec3::new(-horizontal_vel.z, 0.0, horizontal_vel.x).normalize_or_zero()) * 0.04).clamp(-0.5, 0.5);

                            let target_rot = Quat::from_rotation_y(yaw)
                                * Quat::from_rotation_x(forward_tilt)
                                * Quat::from_rotation_z(bank_tilt);
                            drone.rotation = drone.rotation.slerp(target_rot, (dt * 9.0).min(1.0));
                        }

                        // Check gate center floating cube hit
                        for gate in &mut self.gates {
                            let diff = drone.position - gate.position;
                            if diff.length_squared() < 3.2 {
                                gate.hit_timer = 1.0;
                            }
                        }

                        // Check collision with ground
                        if drone.position.y < 0.4 && drone.velocity.y < -1.0 {
                            crash_now = true;
                        }

                        // Check collision with towers
                        for tower in &self.towers {
                            let dx = drone.position.x - tower.position.x;
                            let dz = drone.position.z - tower.position.z;
                            let dist_xz = (dx * dx + dz * dz).sqrt();
                            if dist_xz < tower.radius + 0.45 && drone.position.y < tower.height {
                                crash_now = true;
                                break;
                            }
                        }
                    }

                    DroneState::Takeoff {
                        elapsed,
                        duration,
                        pad_pos,
                        target_alt,
                    } => {
                        *elapsed += dt;
                        let progress = (*elapsed / *duration).clamp(0.0, 1.0);
                        let smooth = progress * progress * (3.0 - 2.0 * progress);
                        drone.position = Vec3::new(
                            pad_pos.x,
                            pad_pos.y + smooth * *target_alt,
                            pad_pos.z,
                        );
                        drone.velocity = Vec3::new(0.0, *target_alt / *duration, 0.0);
                        drone.rotation = Quat::from_rotation_y(progress * TAU * 2.0);

                        if progress >= 1.0 {
                            let initial_target = self.rng.random_point_in_arena();
                            new_state = Some(DroneState::Flying {
                                target: initial_target,
                                waypoint_timer: 0.0,
                            });
                        }
                    }

                    DroneState::Crashing {
                        elapsed,
                        duration,
                        debris_pos,
                        debris_vel,
                        debris_rot,
                        debris_ang_vel,
                    } => {
                        *elapsed += dt;
                        for d in 0..DEBRIS_COUNT {
                            debris_vel[d].y -= 9.8 * dt;
                            debris_pos[d] += debris_vel[d] * dt;
                            if debris_pos[d].y < 0.15 {
                                debris_pos[d].y = 0.15;
                                debris_vel[d].y = -debris_vel[d].y * 0.45;
                                debris_vel[d].x *= 0.75;
                                debris_vel[d].z *= 0.75;
                            }
                            debris_rot[d] = Quat::from_scaled_axis(debris_ang_vel[d] * dt) * debris_rot[d];
                        }

                        if *elapsed >= *duration {
                            let next_pad = (i + self.rng.next_u32() as usize) % pad_count;
                            new_state = Some(DroneState::Respawning {
                                timer: self.rng.range(0.8, 2.0),
                                pad_index: next_pad,
                            });
                        }
                    }

                    DroneState::Respawning { timer, pad_index } => {
                        *timer -= dt;
                        let pad = &self.pads[*pad_index];
                        let offset_x = ((i % 8) as f32 - 3.5) * 0.5;
                        let offset_z = (((i / 8) % 8) as f32 - 3.5) * 0.5;
                        drone.position = Vec3::new(
                            pad.position.x + offset_x,
                            0.4,
                            pad.position.z + offset_z,
                        );
                        drone.velocity = Vec3::ZERO;
                        drone.rotation = Quat::IDENTITY;

                        if *timer <= 0.0 {
                            let target_alt = self.rng.range(6.0, 22.0);
                            new_state = Some(DroneState::Takeoff {
                                elapsed: 0.0,
                                duration: self.rng.range(1.6, 2.4),
                                pad_pos: drone.position,
                                target_alt,
                            });
                        }
                    }
                }
            }

            if crash_now {
                self.drones[i].trigger_crash(&mut self.rng);
            } else if let Some(state) = new_state {
                self.drones[i].state = state;
            }
        }

        for idx in crash_list {
            if self.drones[idx].is_active() {
                self.drones[idx].trigger_crash(&mut self.rng);
            }
        }
    }
}

impl Scene for DroneDemoScene {
    fn initialize(&mut self, renderer: &mut Renderer) -> Result<(), RendererError> {
        self.meshes = Some(SceneMeshes {
            cube: renderer.register_mesh(MeshData::cube())?,
            plane: renderer.register_mesh(MeshData::plane())?,
            cylinder: renderer.register_mesh(MeshData::cylinder(16))?,
            sphere: renderer.register_mesh(MeshData::sphere(10, 14))?,
        });

        // Load the 2K Barnaslingan outdoor HDR skybox image
        const SKYBOX_BYTES: &[u8] =
            include_bytes!("../assets/barnaslingan_02_2k.hdr");
        if let Err(err) = renderer.set_skybox_from_image_bytes(SKYBOX_BYTES) {
            eprintln!("Warning: Failed to load skybox image: {err}");
        }

        Ok(())
    }

    fn frame(&mut self, elapsed_seconds: f32, width: u32, height: u32) -> &Frame {
        let dt = if self.last_time == 0.0 {
            0.016
        } else {
            (elapsed_seconds - self.last_time).clamp(0.001, 0.066)
        };
        self.last_time = elapsed_seconds;

        self.update_simulation(dt);

        let meshes = self
            .meshes
            .as_ref()
            .expect("scene is initialized before rendering");

        self.frame.begin();

        // 1. Primary Display View (Interactable Orbit Camera with Mouse Drag)
        if !self.mouse_pressed && !self.is_user_interacting {
            self.camera_yaw += dt * 0.08;
        }

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
                near: 0.2,
                far: 250.0,
            },
            width,
            height,
            outputs: ViewOutputs::COLOR,
        });

        // 2. Draw Environment
        // Arena Floor
        self.frame.draw(RenderPrimitive {
            mesh: meshes.plane,
            transform: Mat4::from_scale_rotation_translation(
                Vec3::new(ARENA_HALF_SIZE * 2.2, 1.0, ARENA_HALF_SIZE * 2.2),
                Quat::IDENTITY,
                Vec3::ZERO,
            ),
            color: [0.015, 0.018, 0.024, 1.0],
            object_id: 1,
        });
        // Launch / Recharge Pads (Square Design)
        for (idx, pad) in self.pads.iter().enumerate() {
            let obj_id = 3 + idx as u32;

            // Main Pad Surface Slab (Square)
            self.frame.draw(RenderPrimitive {
                mesh: meshes.cube,
                transform: Mat4::from_scale_rotation_translation(
                    Vec3::new(pad.size, 0.12, pad.size),
                    Quat::IDENTITY,
                    pad.position,
                ),
                color: pad.color,
                object_id: obj_id,
            });

            // Inner Target Square
            self.frame.draw(RenderPrimitive {
                mesh: meshes.cube,
                transform: Mat4::from_scale_rotation_translation(
                    Vec3::new(pad.size * 0.65, 0.16, pad.size * 0.65),
                    Quat::IDENTITY,
                    pad.position + Vec3::new(0.0, 0.04, 0.0),
                ),
                color: [0.10, 0.65, 0.95, 1.0],
                object_id: obj_id,
            });

            // Center Target Square Inlay
            self.frame.draw(RenderPrimitive {
                mesh: meshes.cube,
                transform: Mat4::from_scale_rotation_translation(
                    Vec3::new(pad.size * 0.28, 0.19, pad.size * 0.28),
                    Quat::IDENTITY,
                    pad.position + Vec3::new(0.0, 0.07, 0.0),
                ),
                color: [0.02, 0.03, 0.04, 1.0],
                object_id: obj_id,
            });

            // 4 Corner Square Warning Pads
            let half = pad.size * 0.42;
            for (cx, cz) in [(-half, -half), (half, -half), (-half, half), (half, half)] {
                self.frame.draw(RenderPrimitive {
                    mesh: meshes.cube,
                    transform: Mat4::from_scale_rotation_translation(
                        Vec3::new(0.5, 0.18, 0.5),
                        Quat::IDENTITY,
                        pad.position + Vec3::new(cx, 0.06, cz),
                    ),
                    color: [1.0, 0.80, 0.10, 1.0],
                    object_id: obj_id,
                });
            }
        }

        // City Obstacle Towers
        for (idx, tower) in self.towers.iter().enumerate() {
            let obj_id = 20 + idx as u32;

            // Main Tower Body (Cylinder)
            self.frame.draw(RenderPrimitive {
                mesh: meshes.cylinder,
                transform: Mat4::from_scale_rotation_translation(
                    Vec3::new(tower.radius * 2.0, tower.height, tower.radius * 2.0),
                    Quat::IDENTITY,
                    tower.position + Vec3::new(0.0, tower.height * 0.5, 0.0),
                ),
                color: tower.color,
                object_id: obj_id,
            });

            // Tower Base Flange / Trim
            self.frame.draw(RenderPrimitive {
                mesh: meshes.cylinder,
                transform: Mat4::from_scale_rotation_translation(
                    Vec3::new(tower.radius * 2.4, 0.8, tower.radius * 2.4),
                    Quat::IDENTITY,
                    tower.position + Vec3::new(0.0, 0.4, 0.0),
                ),
                color: tower.color,
                object_id: obj_id,
            });

            // Tower Top Platform / Crown
            self.frame.draw(RenderPrimitive {
                mesh: meshes.cylinder,
                transform: Mat4::from_scale_rotation_translation(
                    Vec3::new(tower.radius * 2.2, 0.6, tower.radius * 2.2),
                    Quat::IDENTITY,
                    tower.position + Vec3::new(0.0, tower.height + 0.3, 0.0),
                ),
                color: tower.color,
                object_id: obj_id,
            });

            // Glowing Beacon Orb
            let turret_pos = tower.position + Vec3::new(0.0, tower.height + 0.8, 0.0);
            let aim_norm = tower.aim_dir.normalize_or_zero();

            // Turret Swivel Base Platform
            self.frame.draw(RenderPrimitive {
                mesh: meshes.cylinder,
                transform: Mat4::from_scale_rotation_translation(
                    Vec3::new(2.4, 0.5, 2.4),
                    Quat::IDENTITY,
                    turret_pos - Vec3::new(0.0, 0.3, 0.0),
                ),
                color: [0.15, 0.17, 0.22, 1.0],
                object_id: obj_id,
            });

            // Turret Housing Pod (Sphere)
            self.frame.draw(RenderPrimitive {
                mesh: meshes.sphere,
                transform: Mat4::from_scale_rotation_translation(
                    Vec3::new(1.6, 1.2, 1.6),
                    Quat::IDENTITY,
                    turret_pos,
                ),
                color: [0.10, 0.12, 0.16, 1.0],
                object_id: obj_id,
            });

            // Turret Heavy Cannon Barrels (Dual Barrels)
            let barrel_rot = Quat::from_rotation_arc(Vec3::Y, aim_norm);
            let right_dir = aim_norm.cross(Vec3::Y).normalize_or_zero();

            for side in [-0.55, 0.55] {
                let barrel_offset = right_dir * side + aim_norm * 1.5;
                self.frame.draw(RenderPrimitive {
                    mesh: meshes.cylinder,
                    transform: Mat4::from_scale_rotation_translation(
                        Vec3::new(0.30, 3.2, 0.30),
                        barrel_rot,
                        turret_pos + barrel_offset,
                    ),
                    color: [0.22, 0.25, 0.30, 1.0],
                    object_id: obj_id,
                });
            }

            // Glowing Turret Sensor Eye
            let blink = ((elapsed_seconds * 4.0 + idx as f32 * 0.8).sin() * 0.5 + 0.5).max(0.3);
            self.frame.draw(RenderPrimitive {
                mesh: meshes.sphere,
                transform: Mat4::from_scale_rotation_translation(
                    Vec3::splat(0.55),
                    Quat::IDENTITY,
                    turret_pos + aim_norm * 0.9 + Vec3::new(0.0, 0.35, 0.0),
                ),
                color: [
                    tower.beacon_color[0] * blink,
                    tower.beacon_color[1] * blink,
                    tower.beacon_color[2] * blink,
                    1.0,
                ],
                object_id: obj_id,
            });
        }

        // Active Plasma Turret Projectiles
        for proj in &self.projectiles {
            let vel_dir = proj.velocity.normalize_or_zero();
            let tracer_rot = Quat::from_rotation_arc(Vec3::Y, vel_dir);
            // Glowing energy tracer bolt
            self.frame.draw(RenderPrimitive {
                mesh: meshes.cylinder,
                transform: Mat4::from_scale_rotation_translation(
                    Vec3::new(0.24, 2.2, 0.24),
                    tracer_rot,
                    proj.position,
                ),
                color: proj.color,
                object_id: 80,
            });
            // Brilliant glowing plasma head
            self.frame.draw(RenderPrimitive {
                mesh: meshes.sphere,
                transform: Mat4::from_scale_rotation_translation(
                    Vec3::splat(0.48),
                    Quat::IDENTITY,
                    proj.position + vel_dir * 1.1,
                ),
                color: [1.0, 1.0, 1.0, 1.0],
                object_id: 80,
            });
        }

        // Sky Waypoint Square Gates
        // Sky Waypoint Square Gates (Interactive Color on Center Cube Hit)
        for (idx, gate) in self.gates.iter().enumerate() {
            let obj_id = 50 + idx as u32;
            let rot = Quat::from_rotation_y(gate.yaw);
            let half = gate.size;
            let thickness = 0.35;
            let length = half * 2.0 + thickness;

            // When hit, transition gate color to vivid neon green
            let active_color = if gate.hit_timer > 0.0 {
                let t = gate.hit_timer.min(1.0);
                let neon_green = [0.15, 0.98, 0.35, 1.0];
                [
                    gate.color[0] + (neon_green[0] - gate.color[0]) * t,
                    gate.color[1] + (neon_green[1] - gate.color[1]) * t,
                    gate.color[2] + (neon_green[2] - gate.color[2]) * t,
                    1.0,
                ]
            } else {
                gate.color
            };

            // Top frame strut
            self.frame.draw(RenderPrimitive {
                mesh: meshes.cube,
                transform: Mat4::from_scale_rotation_translation(
                    Vec3::new(thickness, thickness, length),
                    rot,
                    gate.position + rot * Vec3::new(0.0, half, 0.0),
                ),
                color: active_color,
                object_id: obj_id,
            });

            // Bottom frame strut
            self.frame.draw(RenderPrimitive {
                mesh: meshes.cube,
                transform: Mat4::from_scale_rotation_translation(
                    Vec3::new(thickness, thickness, length),
                    rot,
                    gate.position + rot * Vec3::new(0.0, -half, 0.0),
                ),
                color: active_color,
                object_id: obj_id,
            });

            // Left vertical strut
            self.frame.draw(RenderPrimitive {
                mesh: meshes.cube,
                transform: Mat4::from_scale_rotation_translation(
                    Vec3::new(thickness, length, thickness),
                    rot,
                    gate.position + rot * Vec3::new(0.0, 0.0, -half),
                ),
                color: active_color,
                object_id: obj_id,
            });

            // Right vertical strut
            self.frame.draw(RenderPrimitive {
                mesh: meshes.cube,
                transform: Mat4::from_scale_rotation_translation(
                    Vec3::new(thickness, length, thickness),
                    rot,
                    gate.position + rot * Vec3::new(0.0, 0.0, half),
                ),
                color: active_color,
                object_id: obj_id,
            });

            // 4 Corner Markers / Glowing Joint Caps
            for (cy, cz) in [(-half, -half), (-half, half), (half, -half), (half, half)] {
                self.frame.draw(RenderPrimitive {
                    mesh: meshes.cube,
                    transform: Mat4::from_scale_rotation_translation(
                        Vec3::splat(thickness * 1.6),
                        rot,
                        gate.position + rot * Vec3::new(0.0, cy, cz),
                    ),
                    color: [
                        active_color[0] * 1.3,
                        active_color[1] * 1.3,
                        active_color[2] * 1.3,
                        1.0,
                    ],
                    object_id: obj_id,
                });
            }

            // Glowing Gate Center Floating Diamond/Cube
            let diamond_rot = rot
                * Quat::from_rotation_y(elapsed_seconds * 2.0)
                * Quat::from_rotation_z(FRAC_PI_4);
            let center_color = if gate.hit_timer > 0.0 {
                [0.25, 1.0, 0.45, 1.0] // Bright Green on hit
            } else {
                [0.95, 0.95, 0.95, 1.0] // Luminous White default
            };
            self.frame.draw(RenderPrimitive {
                mesh: meshes.cube,
                transform: Mat4::from_scale_rotation_translation(
                    Vec3::splat(0.65),
                    diamond_rot,
                    gate.position,
                ),
                color: center_color,
                object_id: obj_id,
            });
        }

        // 4. Draw 100 Drones & Debris
        for drone in &self.drones {
            match &drone.state {
                DroneState::Flying { .. } | DroneState::Takeoff { .. } => {
                    let obj_id = drone.id;
                    let pos = drone.position;
                    let rot = drone.rotation;

                    // Central Cockpit Pod (Sphere)
                    self.frame.draw(RenderPrimitive {
                        mesh: meshes.sphere,
                        transform: Mat4::from_scale_rotation_translation(
                            Vec3::new(0.55, 0.35, 0.65),
                            rot,
                            pos,
                        ),
                        color: [0.15, 0.17, 0.22, 1.0],
                        object_id: obj_id,
                    });

                    // Top Aero Canopy Shell (Colored)
                    self.frame.draw(RenderPrimitive {
                        mesh: meshes.cube,
                        transform: Mat4::from_scale_rotation_translation(
                            Vec3::new(0.40, 0.18, 0.48),
                            rot,
                            pos + rot * Vec3::new(0.0, 0.12, 0.0),
                        ),
                        color: drone.color,
                        object_id: obj_id,
                    });

                    // Front Sensor Camera / Lens
                    self.frame.draw(RenderPrimitive {
                        mesh: meshes.sphere,
                        transform: Mat4::from_scale_rotation_translation(
                            Vec3::splat(0.22),
                            rot,
                            pos + rot * Vec3::new(0.0, 0.02, 0.35),
                        ),
                        color: [0.1, 0.9, 1.0, 1.0],
                        object_id: obj_id,
                    });

                    // Flashing Top Beacon
                    let beacon_flash = ((elapsed_seconds * 6.0 + drone.id as f32).sin() * 0.5 + 0.5).max(0.2);
                    self.frame.draw(RenderPrimitive {
                        mesh: meshes.sphere,
                        transform: Mat4::from_scale_rotation_translation(
                            Vec3::splat(0.18),
                            rot,
                            pos + rot * Vec3::new(0.0, 0.22, -0.08),
                        ),
                        color: [
                            drone.beacon_color[0] * beacon_flash,
                            drone.beacon_color[1] * beacon_flash,
                            drone.beacon_color[2] * beacon_flash,
                            1.0,
                        ],
                        object_id: obj_id,
                    });

                    // 4 Diagonal Rotor Arms (X-Configuration)
                    let arm_len = 0.85;
                    let arm_angles = [FRAC_PI_4, 3.0 * FRAC_PI_4, 5.0 * FRAC_PI_4, 7.0 * FRAC_PI_4];
                    for (arm_idx, &angle) in arm_angles.iter().enumerate() {
                        let arm_dir = Vec3::new(angle.cos(), 0.0, angle.sin());
                        let arm_pos = arm_dir * (arm_len * 0.5);

                        // Strut Cylinder
                        self.frame.draw(RenderPrimitive {
                            mesh: meshes.cylinder,
                            transform: Mat4::from_scale_rotation_translation(
                                Vec3::new(0.06, arm_len, 0.06),
                                rot * Quat::from_rotation_y(-angle + FRAC_PI_2) * Quat::from_rotation_z(FRAC_PI_2),
                                pos + rot * arm_pos,
                            ),
                            color: [0.12, 0.14, 0.18, 1.0],
                            object_id: obj_id,
                        });

                        // Motor Mount
                        let tip_pos = arm_dir * arm_len;
                        self.frame.draw(RenderPrimitive {
                            mesh: meshes.cylinder,
                            transform: Mat4::from_scale_rotation_translation(
                                Vec3::new(0.16, 0.14, 0.16),
                                rot,
                                pos + rot * (tip_pos + Vec3::new(0.0, 0.05, 0.0)),
                            ),
                            color: [0.22, 0.25, 0.30, 1.0],
                            object_id: obj_id,
                        });

                        // Spinning Propeller / Rotor Blade
                        let spin_dir = if arm_idx % 2 == 0 { 1.0 } else { -1.0 };
                        let blade_spin = Quat::from_rotation_y(drone.propeller_angle * spin_dir + arm_idx as f32);
                        self.frame.draw(RenderPrimitive {
                            mesh: meshes.cube,
                            transform: Mat4::from_scale_rotation_translation(
                                Vec3::new(0.55, 0.02, 0.08),
                                rot * blade_spin,
                                pos + rot * (tip_pos + Vec3::new(0.0, 0.14, 0.0)),
                            ),
                            color: [0.85, 0.90, 0.95, 1.0],
                            object_id: obj_id,
                        });
                    }
                }

                DroneState::Crashing {
                    elapsed,
                    duration,
                    debris_pos,
                    debris_rot,
                    ..
                } => {
                    let obj_id = drone.id;
                    let progress = (elapsed / duration).clamp(0.0, 1.0);
                    let fade = 1.0 - progress;

                    // Draw active tumbling debris fragments
                    for d in 0..DEBRIS_COUNT {
                        let scale = (0.28 * fade).max(0.08);
                        let mesh = if d % 2 == 0 { meshes.cube } else { meshes.sphere };
                        let color = if d == 0 {
                            [1.0, 0.6 * fade, 0.1 * fade, 1.0] // Burning spark
                        } else {
                            [drone.color[0] * fade, drone.color[1] * fade, drone.color[2] * fade, 1.0]
                        };

                        self.frame.draw(RenderPrimitive {
                            mesh,
                            transform: Mat4::from_scale_rotation_translation(
                                Vec3::splat(scale),
                                debris_rot[d],
                                debris_pos[d],
                            ),
                            color,
                            object_id: obj_id,
                        });
                    }
                }

                DroneState::Respawning { pad_index, .. } => {
                    // Dim resting drone on launch pad
                    let pad = &self.pads[*pad_index];
                    let offset_x = ((drone.id % 4) as f32 - 1.5) * 0.9;
                    let offset_z = (((drone.id / 4) % 4) as f32 - 1.5) * 0.9;
                    let resting_pos = Vec3::new(pad.position.x + offset_x, 0.22, pad.position.z + offset_z);

                    self.frame.draw(RenderPrimitive {
                        mesh: meshes.cube,
                        transform: Mat4::from_scale_rotation_translation(
                            Vec3::new(0.5, 0.15, 0.5),
                            Quat::IDENTITY,
                            resting_pos,
                        ),
                        color: [drone.color[0] * 0.35, drone.color[1] * 0.35, drone.color[2] * 0.35, 1.0],
                        object_id: drone.id,
                    });
                }
            }
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
                if self.mouse_pressed {
                    self.is_user_interacting = true;
                } else {
                    self.last_mouse_pos = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    if let Some((last_x, last_y)) = self.last_mouse_pos {
                        let dx = (position.x - last_x) as f32;
                        let dy = (position.y - last_y) as f32;
                        let sensitivity = 0.005;
                        self.camera_yaw -= dx * sensitivity;
                        self.camera_pitch =
                            (self.camera_pitch + dy * sensitivity).clamp(0.05, 1.50);
                    }
                    self.last_mouse_pos = Some((position.x, position.y));
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_x, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => (pos.y as f32) * 0.05,
                };
                self.camera_distance = (self.camera_distance + scroll * 2.5).clamp(10.0, 130.0);
                self.is_user_interacting = true;
            }
            _ => {}
        }
    }
}
