use std::{
    cell::RefCell,
    ffi::{c_char, c_void},
    panic::{AssertUnwindSafe, catch_unwind},
    time::Instant,
};

use anyhow::{Context, Result, bail, ensure};
use glam::{Mat4, Quat, Vec3};
use sim_graphics::{
    Camera, Frame, MeshData, ReadbackData, RenderPrimitive, RenderView, Renderer, ViewKey,
    ViewKind, ViewOutputs,
};

const HISTORY: usize = 4;
const ACTIONS: usize = 2;
const MAX_SPEED: f32 = 0.75;
const CONTROL_DT: f32 = 0.05;
const VEHICLE_RADIUS: f32 = 0.30;
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

thread_local! {
    static LAST_ERROR: RefCell<Vec<u8>> = RefCell::new(vec![0]);
}

#[derive(Clone, Copy, Default)]
struct State {
    x: f32,
    z: f32,
    vx: f32,
    vz: f32,
    episode: u64,
    length: u32,
    return_: f32,
}

pub struct RgbEnv {
    n: usize,
    width: u32,
    height: u32,
    max_steps: u32,
    seed: u64,
    frame_bytes: usize,
    history_bytes: usize,
    rgba_frame_bytes: usize,
    states: Vec<State>,
    observations: Vec<u8>,
    final_observations: Vec<u8>,
    rgba_batch: Vec<u8>,
    rewards: Vec<f32>,
    terminated: Vec<f32>,
    truncated: Vec<f32>,
    completed_returns: Vec<f32>,
    completed_lengths: Vec<f32>,
    episode_counts: Vec<u64>,
    current_returns: Vec<f32>,
    current_lengths: Vec<f32>,
    renderer: Renderer,
    cube: sim_graphics::MeshHandle,
    plane: sim_graphics::MeshHandle,
    frame: Frame,
    last_dynamics_ms: f64,
    last_render_ms: f64,
    last_history_ms: f64,
}

impl RgbEnv {
    fn new(n: usize, seed: u64, max_steps: u32, width: u32, height: u32) -> Result<Self> {
        ensure!(n > 0, "environment count must be positive");
        ensure!(max_steps > 0, "max_steps must be positive");
        ensure!(width > 0 && height > 0, "image dimensions must be positive");
        let frame_pixels = usize::try_from(width)
            .and_then(|w| usize::try_from(height).map(|h| w * h))
            .context("image dimensions overflow host usize")?;
        let frame_bytes = frame_pixels
            .checked_mul(3)
            .context("RGB frame size overflow")?;
        let rgba_frame_bytes = frame_pixels
            .checked_mul(4)
            .context("RGBA frame size overflow")?;
        let history_bytes = frame_bytes
            .checked_mul(HISTORY)
            .context("history size overflow")?;
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let mut renderer = pollster::block_on(Renderer::new(&instance, None))?;
        let cube = renderer.register_mesh(MeshData::cube())?;
        let plane = renderer.register_mesh(MeshData::plane())?;
        let mut env = Self {
            n,
            width,
            height,
            max_steps,
            seed,
            frame_bytes,
            history_bytes,
            rgba_frame_bytes,
            states: vec![State::default(); n],
            observations: vec![0; n * history_bytes],
            final_observations: vec![0; n * history_bytes],
            rgba_batch: vec![0; n * rgba_frame_bytes],
            rewards: vec![0.0; n],
            terminated: vec![0.0; n],
            truncated: vec![0.0; n],
            completed_returns: vec![0.0; n],
            completed_lengths: vec![0.0; n],
            episode_counts: vec![0; n],
            current_returns: vec![0.0; n],
            current_lengths: vec![0.0; n],
            renderer,
            cube,
            plane,
            frame: Frame::with_capacity(n * 4, n),
            last_dynamics_ms: 0.0,
            last_render_ms: 0.0,
            last_history_ms: 0.0,
        };
        env.reset_internal(seed)?;
        Ok(env)
    }

    fn reset_internal(&mut self, seed: u64) -> Result<()> {
        self.seed = seed;
        self.rewards.fill(0.0);
        self.terminated.fill(0.0);
        self.truncated.fill(0.0);
        self.completed_returns.fill(0.0);
        self.completed_lengths.fill(0.0);
        self.episode_counts.fill(0);
        for state in &mut self.states {
            *state = State::default();
        }
        self.last_dynamics_ms = 0.0;
        self.last_history_ms = 0.0;
        let render_start = Instant::now();
        self.render_batch()?;
        self.last_render_ms = render_start.elapsed().as_secs_f64() * 1_000.0;
        for env in 0..self.n {
            self.fill_history_from_current(env);
        }
        self.current_returns.fill(0.0);
        self.current_lengths.fill(0.0);
        Ok(())
    }

    fn step_internal(&mut self, actions: &[f32]) -> Result<()> {
        ensure!(
            actions.len() >= self.n * ACTIONS,
            "action buffer is too small"
        );
        self.rewards.fill(0.0);
        self.terminated.fill(0.0);
        self.truncated.fill(0.0);
        self.completed_returns.fill(0.0);
        self.completed_lengths.fill(0.0);

        let dynamics_start = Instant::now();
        let mut done = vec![false; self.n];
        for env in 0..self.n {
            let state = &mut self.states[env];
            let forward = actions[env * ACTIONS].clamp(-1.0, 1.0);
            let lateral = actions[env * ACTIONS + 1].clamp(-1.0, 1.0);
            let target_vx = lateral * MAX_SPEED;
            let target_vz = -forward * MAX_SPEED;
            state.vx += (target_vx - state.vx) * 0.35;
            state.vz += (target_vz - state.vz) * 0.35;
            state.x += state.vx * CONTROL_DT;
            state.z += state.vz * CONTROL_DT;
            state.length += 1;

            let collision = collides(state.x, state.z);
            let success = state.z <= -7.0;
            let timeout = state.length >= self.max_steps;
            let reward = -0.002 + (-state.vz).max(0.0) * 0.08;
            self.rewards[env] = reward
                + if collision {
                    -1.0
                } else if success {
                    1.0
                } else {
                    0.0
                };
            state.return_ += self.rewards[env];
            self.terminated[env] = f32::from(collision || success);
            self.truncated[env] = f32::from(timeout && !collision && !success);
            done[env] = collision || success || timeout;
        }

        self.last_dynamics_ms = dynamics_start.elapsed().as_secs_f64() * 1_000.0;
        self.last_render_ms = 0.0;
        self.last_history_ms = 0.0;
        let render_start = Instant::now();
        self.render_batch()?;
        self.last_render_ms += render_start.elapsed().as_secs_f64() * 1_000.0;
        let history_start = Instant::now();
        for env in 0..self.n {
            self.append_current_frame(env);
            if done[env] {
                let base = env * self.history_bytes;
                self.final_observations[base..base + self.history_bytes]
                    .copy_from_slice(&self.observations[base..base + self.history_bytes]);
                self.completed_returns[env] = self.states[env].return_;
                self.completed_lengths[env] = self.states[env].length as f32;
                self.episode_counts[env] += 1;
                self.states[env] = State {
                    episode: self.states[env].episode + 1,
                    ..State::default()
                };
            }
        }
        self.last_history_ms += history_start.elapsed().as_secs_f64() * 1_000.0;

        if done.iter().any(|value| *value) {
            let render_start = Instant::now();
            self.render_batch()?;
            self.last_render_ms += render_start.elapsed().as_secs_f64() * 1_000.0;
            let history_start = Instant::now();
            for (env, is_done) in done.into_iter().enumerate() {
                if is_done {
                    self.fill_history_from_current(env);
                }
            }
            self.last_history_ms += history_start.elapsed().as_secs_f64() * 1_000.0;
        }
        for env in 0..self.n {
            self.current_returns[env] = self.states[env].return_;
            self.current_lengths[env] = self.states[env].length as f32;
        }
        Ok(())
    }

    fn render_batch(&mut self) -> Result<()> {
        self.frame.begin();
        let grid = (self.n as f32).sqrt().ceil() as usize;
        let spacing = 16.0f32;
        let fov = 60.0f32.to_radians();
        for env in 0..self.n {
            let gx = env % grid;
            let gz = env / grid;
            let origin = Vec3::new(gx as f32 * spacing, 0.0, gz as f32 * spacing);
            let state = self.states[env];
            let vehicle = origin + Vec3::new(state.x, 0.0, state.z);
            self.frame.add_view(RenderView {
                key: ViewKey(env as u64),
                kind: ViewKind::Sensor,
                camera: Camera {
                    eye: vehicle + Vec3::new(0.0, 2.0, 4.5),
                    target: vehicle + Vec3::new(0.0, 1.0, -2.0),
                    up: Vec3::Y,
                    vertical_fov_radians: fov,
                    near: 0.05,
                    far: 14.0,
                },
                width: self.width,
                height: self.height,
                outputs: ViewOutputs::COLOR,
            });
            self.frame.draw(RenderPrimitive {
                mesh: self.plane,
                transform: Mat4::from_scale_rotation_translation(
                    Vec3::new(9.0, 1.0, 13.0),
                    Quat::IDENTITY,
                    origin + Vec3::new(0.0, -0.05, -2.5),
                ),
                color: [0.16, 0.19, 0.22, 1.0],
                object_id: env as u32 * 10 + 1,
            });
            for (index, (position, scale)) in obstacles().into_iter().enumerate() {
                self.frame.draw(RenderPrimitive {
                    mesh: self.cube,
                    transform: Mat4::from_scale_rotation_translation(
                        scale,
                        Quat::IDENTITY,
                        origin + position,
                    ),
                    color: obstacle_color(env, index),
                    object_id: env as u32 * 10 + 2 + index as u32,
                });
            }
        }

        let submission = self.renderer.execute(&self.frame, &[])?;
        for (env, handle) in submission.readbacks.into_iter().enumerate() {
            let data = loop {
                if let Some(data) = self.renderer.poll_readback(handle)? {
                    break data;
                }
                std::thread::yield_now();
            };
            let pixels = match data {
                ReadbackData::Color(pixels) => pixels,
                other => bail!("expected color readback, received {other:?}"),
            };
            ensure!(
                pixels.len() == self.rgba_frame_bytes,
                "renderer returned {} bytes for {}-byte image",
                pixels.len(),
                self.rgba_frame_bytes
            );
            let base = env * self.rgba_frame_bytes;
            self.rgba_batch[base..base + self.rgba_frame_bytes].copy_from_slice(&pixels);
        }
        Ok(())
    }

    fn append_current_frame(&mut self, env: usize) {
        let history_base = env * self.history_bytes;
        self.observations.copy_within(
            history_base + self.frame_bytes..history_base + self.history_bytes,
            history_base,
        );
        let rgba_base = env * self.rgba_frame_bytes;
        let rgb_base = history_base + self.history_bytes - self.frame_bytes;
        for pixel in 0..self.frame_bytes / 3 {
            let source = rgba_base + pixel * 4;
            let target = rgb_base + pixel * 3;
            self.observations[target..target + 3]
                .copy_from_slice(&self.rgba_batch[source..source + 3]);
        }
    }

    fn fill_history_from_current(&mut self, env: usize) {
        self.append_current_frame(env);
        let base = env * self.history_bytes;
        let last = base + self.history_bytes - self.frame_bytes;
        let frame = self.observations[last..last + self.frame_bytes].to_vec();
        for history in 0..HISTORY - 1 {
            let start = base + history * self.frame_bytes;
            self.observations[start..start + self.frame_bytes].copy_from_slice(&frame);
        }
    }

    fn buffer(&mut self, field: i32) -> *mut c_void {
        match field {
            0 => self.observations.as_mut_ptr().cast(),
            1 => self.rewards.as_mut_ptr().cast(),
            2 => self.terminated.as_mut_ptr().cast(),
            3 => self.truncated.as_mut_ptr().cast(),
            4 => self.final_observations.as_mut_ptr().cast(),
            5 => self.completed_returns.as_mut_ptr().cast(),
            6 => self.completed_lengths.as_mut_ptr().cast(),
            7 => self.episode_counts.as_mut_ptr().cast(),
            8 => self.current_returns.as_mut_ptr().cast(),
            9 => self.current_lengths.as_mut_ptr().cast(),
            _ => std::ptr::null_mut(),
        }
    }

    fn checksum(&self) -> u64 {
        let mut checksum = FNV_OFFSET;
        for byte in &self.observations {
            checksum ^= u64::from(*byte);
            checksum = checksum.wrapping_mul(FNV_PRIME);
        }
        checksum
    }
    fn timings(&self, values: &mut [f64]) -> Result<()> {
        ensure!(values.len() >= 3, "timing buffer is too small");
        values[0] = self.last_dynamics_ms;
        values[1] = self.last_render_ms;
        values[2] = self.last_history_ms;
        Ok(())
    }
}

fn obstacles() -> [(Vec3, Vec3); 3] {
    [
        (Vec3::new(-1.4, 0.8, -2.4), Vec3::new(0.9, 1.6, 0.8)),
        (Vec3::new(1.1, 1.1, -4.2), Vec3::new(1.5, 2.2, 0.9)),
        (Vec3::new(-0.1, 0.55, -6.1), Vec3::new(2.5, 1.1, 0.7)),
    ]
}

fn collides(x: f32, z: f32) -> bool {
    obstacles().into_iter().any(|(center, scale)| {
        let half_x = scale.x * 0.5 + VEHICLE_RADIUS;
        let half_z = scale.z * 0.5 + VEHICLE_RADIUS;
        (x - center.x).abs() <= half_x && (z - center.z).abs() <= half_z
    })
}

fn obstacle_color(env: usize, index: usize) -> [f32; 4] {
    let mut value = (env as u32)
        .wrapping_mul(747796405)
        .wrapping_add((index as u32 + 1).wrapping_mul(2891336453));
    value ^= value >> 16;
    value = value.wrapping_mul(2246822519);
    [
        0.25 + (value & 255) as f32 / 510.0,
        0.25 + ((value >> 8) & 255) as f32 / 510.0,
        0.25 + ((value >> 16) & 255) as f32 / 510.0,
        1.0,
    ]
}

fn set_error(error: impl std::fmt::Display) {
    let mut bytes = error.to_string().into_bytes();
    bytes.retain(|byte| *byte != 0);
    bytes.push(0);
    LAST_ERROR.with(|slot| *slot.borrow_mut() = bytes);
}

fn clear_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = vec![0]);
}

fn call<T>(operation: impl FnOnce() -> Result<T>) -> Result<T> {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(result) => result,
        Err(_) => bail!("native RGB environment panicked"),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn triage_rgb_error() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr().cast())
}

#[unsafe(no_mangle)]
pub extern "C" fn triage_rgb_create(
    n: usize,
    seed: u64,
    max_steps: u32,
    width: u32,
    height: u32,
) -> *mut c_void {
    clear_error();
    match call(|| RgbEnv::new(n, seed, max_steps, width, height)) {
        Ok(env) => Box::into_raw(Box::new(env)).cast(),
        Err(error) => {
            set_error(error);
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn triage_rgb_reset(env: *mut c_void, seed: u64) -> i32 {
    clear_error();
    let result = call(|| {
        let env = unsafe { (env as *mut RgbEnv).as_mut() }.context("null RGB environment")?;
        env.reset_internal(seed)
    });
    match result {
        Ok(()) => 0,
        Err(error) => {
            set_error(error);
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn triage_rgb_step(env: *mut c_void, actions: *const f32) -> i32 {
    clear_error();
    let result = call(|| {
        let env = unsafe { (env as *mut RgbEnv).as_mut() }.context("null RGB environment")?;
        let actions = unsafe { std::slice::from_raw_parts(actions, env.n * ACTIONS) };
        env.step_internal(actions)
    });
    match result {
        Ok(()) => 0,
        Err(error) => {
            set_error(error);
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn triage_rgb_timings(env: *mut c_void, values: *mut f64) -> i32 {
    clear_error();
    let result = call(|| {
        let env = unsafe { (env as *mut RgbEnv).as_ref() }.context("null RGB environment")?;
        ensure!(!values.is_null(), "null timing buffer");
        let values = unsafe { std::slice::from_raw_parts_mut(values, 3) };
        env.timings(values)
    });
    match result {
        Ok(()) => 0,
        Err(error) => {
            set_error(error);
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn triage_rgb_buffer(env: *mut c_void, field: i32) -> *mut c_void {
    clear_error();
    let result = call(|| {
        let env = unsafe { (env as *mut RgbEnv).as_mut() }.context("null RGB environment")?;
        Ok(env.buffer(field))
    });
    match result {
        Ok(pointer) => pointer,
        Err(error) => {
            set_error(error);
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn triage_rgb_checksum(env: *mut c_void) -> u64 {
    let result = call(|| {
        let env = unsafe { (env as *mut RgbEnv).as_ref() }.context("null RGB environment")?;
        Ok(env.checksum())
    });
    match result {
        Ok(checksum) => checksum,
        Err(error) => {
            set_error(error);
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn triage_rgb_destroy(env: *mut c_void) -> i32 {
    clear_error();
    if env.is_null() {
        set_error("null RGB environment");
        return -1;
    }
    unsafe {
        drop(Box::from_raw(env as *mut RgbEnv));
    }
    0
}
