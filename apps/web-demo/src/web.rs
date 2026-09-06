mod controls;
mod depth_range;
mod gui;
mod scene;

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use glam::Vec3;
use sim_graphics::{
    Camera, Frame, OutputKind, RenderView, Renderer, ViewKey, ViewKind, ViewOutputs,
};
use sim_trajectory::{CameraMode, GROUND_OBJECT_ID, Recording, TrajectoryScene};
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

const NEAR: f32 = 0.1;
const FAR: f32 = 80.0;
const FOV_DEGREES: f32 = 50.0;
const SENSOR: ViewKey = ViewKey(1);

fn error(message: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&message.to_string())
}

#[wasm_bindgen]
pub async fn create_renderer(canvas: HtmlCanvasElement) -> Result<Engine, JsValue> {
    console_error_panic_hook::set_once();
    let window = web_sys::window().ok_or_else(|| error("This demo requires a browser window."))?;
    let navigator = js_sys::Reflect::get(&window, &"navigator".into())?;
    let gpu = js_sys::Reflect::get(&navigator, &"gpu".into())?;
    if gpu.is_null() || gpu.is_undefined() {
        return Err(error(
            "WebGPU is unavailable. Open this demo in a WebGPU-capable browser on HTTPS or localhost; no WebGL fallback is provided.",
        ));
    }
    Engine::new(canvas).await
}

#[wasm_bindgen]
pub struct Engine {
    instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    canvas: HtmlCanvasElement,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    bindings: Option<wgpu::BindGroup>,
    sensor_size: [u32; 2],
    scenes: [scene::Scene; 2],
    scene: usize,
    frame: Frame,
    healthy: Arc<AtomicBool>,
    controls: controls::Controls,
    gui: gui::Gui,
    depth_range: depth_range::DepthRange,
    trajectory_scene: TrajectoryScene,
    recording: Option<Recording>,
    playback_time: f64,
    environment_index: usize,
}

impl Engine {
    async fn new(canvas: HtmlCanvasElement) -> Result<Self, JsValue> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        // Never select the WebGL backend: metric depth and integer labels are real WebGPU outputs.
        descriptor.backends = wgpu::Backends::BROWSER_WEBGPU;
        let instance = wgpu::Instance::new(descriptor);
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(error)?;
        let mut renderer = Renderer::new(&instance, Some(&surface)).await.map_err(|cause| {
            error(format!("WebGPU initialization failed. Use a WebGPU-capable browser on HTTPS or localhost: {cause}"))
        })?;
        let healthy = Arc::new(AtomicBool::new(true));
        let on_error = healthy.clone();
        renderer
            .device()
            .on_uncaptured_error(Arc::new(move |cause| {
                on_error.store(false, Ordering::Relaxed);
                web_sys::console::error_1(&error(format!("WebGPU error: {cause}")));
            }));
        let on_lost = healthy.clone();
        renderer
            .device()
            .set_device_lost_callback(move |reason, message| {
                on_lost.store(false, Ordering::Relaxed);
                web_sys::console::error_1(&error(format!(
                    "WebGPU device lost ({reason:?}): {message}. Reload to reconnect."
                )));
            });
        let scenes = scene::build(&mut renderer).map_err(error)?;
        let mut trajectory_scene = TrajectoryScene::new();
        trajectory_scene.initialize(&mut renderer).map_err(error)?;
        let device = renderer.device();
        let depth_range = depth_range::DepthRange::new(device);
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("web demo selector settings"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(16),
            },
            count: None,
        }];
        for binding in 1..=3 {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: if binding == 3 {
                        wgpu::TextureSampleType::Uint
                    } else {
                        wgpu::TextureSampleType::Float { filterable: false }
                    },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
        }
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(8),
            },
            count: None,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("web demo sensor selectors"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("web demo compositor"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("web demo compositor"),
            source: wgpu::ShaderSource::Wgsl(include_str!("compositor.wgsl").into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("web demo compositor"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: renderer.color_format(),
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let capabilities = surface.get_capabilities(renderer.adapter());
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: renderer.color_format(),
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: canvas.width().clamp(1, 2048),
            height: canvas.height().clamp(1, 2048),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 1,
            alpha_mode: capabilities
                .alpha_modes
                .first()
                .copied()
                .ok_or_else(|| error("WebGPU surface has no supported alpha mode."))?,
            view_formats: vec![],
        };
        canvas.set_width(config.width);
        canvas.set_height(config.height);
        surface.configure(device, &config);
        let gui = gui::Gui::new(device, renderer.queue(), renderer.color_format());
        if !healthy.load(Ordering::Relaxed) {
            return Err(error("WebGPU failed while creating demo resources."));
        }
        Ok(Self {
            instance,
            surface,
            canvas,
            config,
            renderer,
            pipeline,
            layout,
            uniform,
            bindings: None,
            sensor_size: [0, 0],
            scenes,
            scene: 0,
            frame: Frame::with_capacity(32, 1),
            healthy,
            controls: controls::Controls::new(),
            gui,
            depth_range,
            trajectory_scene,
            recording: None,
            playback_time: 0.0,
            environment_index: 0,
        })
    }

    fn render_frame(&mut self, width: u32, height: u32, pixel_ratio: f32) -> Result<(), JsValue> {
        let (mode, yaw, pitch, distance, scene) = (
            self.controls.mode,
            self.controls.yaw,
            self.controls.pitch,
            self.controls.distance,
            self.controls.scene,
        );
        if !self.healthy.load(Ordering::Relaxed) {
            return Err(error("WebGPU device is unavailable. Reload to reconnect."));
        }
        if width == 0
            || height == 0
            || width > 2048
            || height > 2048
            || mode > 3
            || scene > 1
            || !yaw.is_finite()
            || !pitch.is_finite()
            || !distance.is_finite()
        {
            return Err(error(
                "Invalid render dimensions, mode, scene, or camera parameters.",
            ));
        }
        if [width, height] != [self.config.width, self.config.height] {
            self.config.width = width;
            self.config.height = height;
            self.canvas.set_width(width);
            self.canvas.set_height(height);
            self.surface.configure(self.renderer.device(), &self.config);
        }
        let mut acquired = self.surface.get_current_texture();
        if matches!(acquired, wgpu::CurrentSurfaceTexture::Lost) {
            self.surface = self
                .instance
                .create_surface(wgpu::SurfaceTarget::Canvas(self.canvas.clone()))
                .map_err(error)?;
            self.surface.configure(self.renderer.device(), &self.config);
            acquired = self.surface.get_current_texture();
        } else if matches!(acquired, wgpu::CurrentSurfaceTexture::Outdated) {
            self.surface.configure(self.renderer.device(), &self.config);
            acquired = self.surface.get_current_texture();
        }
        let (surface_texture, suboptimal) = match acquired {
            wgpu::CurrentSurfaceTexture::Success(texture) => (texture, false),
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => (texture, true),
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                return Err(error(
                    "WebGPU surface recovery failed. Reload to reconnect.",
                ));
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Err(error(
                    "WebGPU surface is temporarily unavailable. Reload to reconnect.",
                ));
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(error("WebGPU surface validation failed."));
            }
        };
        let panel_width = if mode == 3 && width >= height {
            width as f32 / 3.0
        } else {
            width as f32
        };
        let panel_height = if mode == 3 && width < height {
            height as f32 / 3.0
        } else {
            height as f32
        };
        // Bound retained pool allocations during continuous browser resizing.
        // Five 4:3 tiers cover display demand without accumulating a texture set per pixel size.
        let demand = (panel_width / 4.0)
            .min(panel_height / 3.0)
            .ceil()
            .clamp(1.0, 240.0) as u32;
        let units = [64, 96, 128, 192, 240]
            .into_iter()
            .find(|&units| units >= demand)
            .unwrap_or(240);
        let dimensions = [units * 4, units * 3];
        let selected = &self.scenes[scene as usize];
        let pitch = pitch.clamp(0.08, 1.45);
        let distance = distance.clamp(4.0, 50.0);
        let offset = Vec3::new(
            yaw.sin() * pitch.cos(),
            pitch.sin(),
            yaw.cos() * pitch.cos(),
        ) * distance;
        self.frame.begin();
        let camera =
            if let Some(recording) = self.recording.as_ref().filter(|_| self.controls.trajectory) {
                let snapshot = recording.sample(self.playback_time);
                self.trajectory_scene.draw(&mut self.frame, snapshot);
                let mode = if self.controls.mounted {
                    CameraMode::Mounted {
                        environment_id: recording.header.environment_ids[self.environment_index],
                    }
                } else {
                    CameraMode::Orbit {
                        azimuth: yaw,
                        elevation: pitch,
                        distance,
                    }
                };
                self.trajectory_scene
                    .camera(&recording.header, snapshot, mode)
            } else {
                for &primitive in &selected.primitives {
                    self.frame.draw(primitive);
                }
                Camera {
                    eye: selected.target + offset,
                    target: selected.target,
                    up: Vec3::Y,
                    vertical_fov_radians: FOV_DEGREES.to_radians(),
                    near: NEAR,
                    far: FAR,
                }
            };
        self.frame.add_view(RenderView {
            key: SENSOR,
            kind: ViewKind::Sensor,
            camera,
            width: dimensions[0],
            height: dimensions[1],
            outputs: ViewOutputs::COLOR | ViewOutputs::DEPTH | ViewOutputs::OBJECT_ID,
        });
        self.renderer.execute_gpu(&self.frame, &[]).map_err(error)?;
        // The one-view, fixed-output graph reuses exactly these textures until dimensions change.
        // Invalidate before rebinding a different pool allocation; no per-frame texture/view churn.
        if dimensions != self.sensor_size {
            self.bindings = None;
        }
        if self.bindings.is_none() {
            let view = |kind| {
                self.renderer
                    .output_texture(SENSOR, kind)
                    .map(|texture| texture.create_view(&Default::default()))
                    .ok_or_else(|| error("Renderer did not produce the requested sensor output."))
            };
            let color = view(OutputKind::Color)?;
            let depth = view(OutputKind::Depth)?;
            let ids = view(OutputKind::ObjectId)?;
            self.depth_range.bind(self.renderer.device(), &depth, &ids);
            self.bindings = Some(self.renderer.device().create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("web demo sensor outputs"),
                    layout: &self.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&color),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&depth),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&ids),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.depth_range.buffer().as_entire_binding(),
                        },
                    ],
                },
            ));
        }
        let uniforms = [
            width as f32,
            height as f32,
            mode as f32,
            if self.config.format.is_srgb() {
                1.0
            } else {
                0.0
            },
        ];
        self.renderer
            .queue()
            .write_buffer(&self.uniform, 0, bytemuck::cast_slice(&uniforms));
        let target = surface_texture.texture.create_view(&Default::default());
        let mut encoder =
            self.renderer
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("web demo presentation"),
                });
        if mode == 1 || mode == 3 {
            self.depth_range
                .encode(self.renderer.queue(), &mut encoder, dimensions);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("web demo selectors"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(
                0,
                self.bindings.as_ref().expect("sensor bindings initialized"),
                &[],
            );
            pass.draw(0..3, 0..1);
        }
        self.gui.render(
            self.renderer.device(),
            self.renderer.queue(),
            &mut encoder,
            &target,
            width,
            height,
            pixel_ratio,
            self.controls.buttons(),
        );
        self.renderer.queue().submit(Some(encoder.finish()));
        self.renderer.present(surface_texture);
        if suboptimal {
            self.surface.configure(self.renderer.device(), &self.config);
        }
        self.sensor_size = dimensions;
        self.scene = scene as usize;
        Ok(())
    }
}

#[wasm_bindgen]
impl Engine {
    /// Draw the scene and its controls. Dimensions are backing pixels; pixel_ratio
    /// is backing pixels per CSS pixel. Supply elapsed seconds, or zero for a still frame.
    pub fn render(
        &mut self,
        width: u32,
        height: u32,
        pixel_ratio: f32,
        delta_seconds: f32,
    ) -> Result<(), JsValue> {
        if width == 0
            || height == 0
            || width > 2048
            || height > 2048
            || !pixel_ratio.is_finite()
            || pixel_ratio <= 0.0
            || !delta_seconds.is_finite()
            || delta_seconds < 0.0
        {
            return Err(error(
                "Invalid canvas dimensions, pixel ratio, or frame interval.",
            ));
        }
        if self.controls.trajectory && self.controls.playing {
            if let Some(recording) = &self.recording {
                let end = recording
                    .snapshots
                    .last()
                    .expect("validated recording")
                    .time_seconds;
                self.playback_time = (self.playback_time + f64::from(delta_seconds)).min(end);
                if self.playback_time >= end {
                    self.controls.playing = false;
                }
            }
        }
        self.controls.advance(delta_seconds);
        self.controls.layout(width, height, pixel_ratio);
        self.render_frame(width, height, pixel_ratio)
    }

    /// UI coordinates are canvas backing pixels. True consumes the pointer for UI.
    pub fn pointer_down(&mut self, x: f32, y: f32) -> bool {
        x.is_finite() && y.is_finite() && self.controls.pointer_down(x, y)
    }

    pub fn pointer_move(&mut self, x: f32, y: f32) -> bool {
        x.is_finite() && y.is_finite() && self.controls.pointer_move(x, y)
    }

    pub fn pointer_up(&mut self, x: f32, y: f32) -> bool {
        if !x.is_finite() || !y.is_finite() {
            return self.controls.pointer_cancel();
        }
        let was_playing = self.controls.playing;
        let consumed = self.controls.pointer_up(x, y);
        if !was_playing && self.controls.playing {
            self.play();
        }
        consumed
    }

    pub fn pointer_cancel(&mut self) -> bool {
        self.controls.pointer_cancel()
    }

    /// Camera drag deltas are CSS pixels, independently of display resolution.
    pub fn orbit_by(&mut self, dx: f32, dy: f32) -> Result<(), JsValue> {
        if !dx.is_finite() || !dy.is_finite() {
            return Err(error("Camera drag must be finite."));
        }
        self.controls.orbit_by(dx, dy);
        if dx != 0.0 || dy != 0.0 {
            self.controls.mounted = false;
        }
        Ok(())
    }

    /// Multipliers above one zoom out; below one zoom in.
    pub fn zoom(&mut self, factor: f32) -> Result<(), JsValue> {
        if !factor.is_finite() || factor <= 0.0 {
            return Err(error("Zoom multiplier must be finite and positive."));
        }
        self.controls.zoom(factor);
        Ok(())
    }

    pub fn key(&mut self, key: &str) -> bool {
        if self.controls.trajectory {
            match key {
                "Home" => {
                    self.restart();
                    return true;
                }
                "[" => {
                    let _ = self.seek(self.playback_time - 1.0);
                    return true;
                }
                "]" => {
                    let _ = self.seek(self.playback_time + 1.0);
                    return true;
                }
                "n" | "N" => {
                    if let Some(recording) = &self.recording {
                        self.environment_index =
                            (self.environment_index + 1) % recording.header.environment_ids.len();
                    }
                    return true;
                }
                " " if !self.controls.playing => {
                    self.play();
                    return true;
                }
                _ => {}
            }
        }
        self.controls.key(key)
    }

    pub fn is_animating(&self) -> bool {
        self.controls.auto_orbit || (self.controls.trajectory && self.controls.playing)
    }

    pub fn mode(&self) -> u32 {
        self.controls.mode
    }

    pub fn scene_index(&self) -> u32 {
        self.controls.scene
    }

    pub fn set_mode(&mut self, mode: u32) -> Result<(), JsValue> {
        if mode > 3 {
            return Err(error("Sensor mode must be 0 through 3."));
        }
        self.controls.set_mode(mode);
        Ok(())
    }

    pub fn set_scene(&mut self, scene: u32) -> Result<(), JsValue> {
        if scene > 1 {
            return Err(error("Scene must be 0 or 1."));
        }
        self.controls.set_scene(scene);
        Ok(())
    }

    pub fn reset_view(&mut self) {
        self.controls.reset();
    }

    /// Parse completely before replacing the active scene. Loading starts paused.
    pub fn load_trajectory(&mut self, ndjson: &str) -> Result<(), JsValue> {
        let recording = Recording::parse(ndjson).map_err(error)?;
        self.playback_time = recording.snapshots[0].time_seconds;
        self.environment_index = 0;
        self.recording = Some(recording);
        self.controls.reset();
        self.controls.trajectory = true;
        self.controls.playing = false;
        Ok(())
    }

    /// Seek to an absolute recording timestamp, clamped to the recorded range.
    pub fn seek(&mut self, time_seconds: f64) -> Result<(), JsValue> {
        if !time_seconds.is_finite() {
            return Err(error("Playback timestamp must be finite."));
        }
        let recording = self
            .recording
            .as_ref()
            .filter(|_| self.controls.trajectory)
            .ok_or_else(|| error("Load a trajectory before seeking."))?;
        self.playback_time = time_seconds.clamp(
            recording.snapshots[0].time_seconds,
            recording
                .snapshots
                .last()
                .expect("validated recording")
                .time_seconds,
        );
        Ok(())
    }

    pub fn play(&mut self) {
        if self.controls.trajectory {
            if let Some(recording) = &self.recording {
                if self.playback_time
                    >= recording
                        .snapshots
                        .last()
                        .expect("validated recording")
                        .time_seconds
                {
                    self.playback_time = recording.snapshots[0].time_seconds;
                }
                self.controls.playing = true;
            }
        }
    }

    pub fn pause(&mut self) {
        self.controls.playing = false;
    }

    pub fn restart(&mut self) {
        if self.controls.trajectory {
            if let Some(recording) = &self.recording {
                self.playback_time = recording.snapshots[0].time_seconds;
            }
        }
    }

    pub fn set_camera_mode(&mut self, mode: &str) -> Result<(), JsValue> {
        self.controls.mounted = match mode {
            "orbit" => false,
            "mounted" => true,
            _ => return Err(error("Camera mode must be orbit or mounted.")),
        };
        self.controls.auto_orbit = false;
        Ok(())
    }

    pub fn info(&self) -> String {
        let scene = &self.scenes[self.scene];
        let mut info = serde_json::json!({
            "backend": "WebGPU",
            "depth_visualization": "relative_visible_range",
            "playback": { "active": false, "playing": false },
            "view": {
                "mode": self.controls.mode, "scene": self.controls.scene,
                "yaw": self.controls.yaw, "pitch": self.controls.pitch,
                "distance": self.controls.distance, "auto_orbit": self.controls.auto_orbit,
            },
            "scene_name": scene.name,
            "object_count": scene.objects.len(),
            "primitive_count": scene.primitives.len(),
            "camera": {
                "near": NEAR, "far": FAR, "fov_vertical_degrees": FOV_DEGREES,
                "width": self.sensor_size[0], "height": self.sensor_size[1],
            },
            "objects": scene.objects.iter().map(|object| serde_json::json!({
                "id": object.id, "name": object.name, "color": scene::palette(object.id),
            })).collect::<Vec<_>>(),
        });
        if let Some(recording) = self.recording.as_ref().filter(|_| self.controls.trajectory) {
            let snapshot = recording.sample(self.playback_time);
            info["scene_name"] = "trajectory".into();
            info["object_count"] = (1 + snapshot.vehicles.len() * 2).into();
            info["primitive_count"] = self.frame.primitives().len().into();
            info["playback"] = serde_json::json!({
                "active": true,
                "playing": self.controls.playing,
                "time_seconds": self.playback_time,
                "start_seconds": recording.snapshots[0].time_seconds,
                "end_seconds": recording.snapshots.last().expect("validated recording").time_seconds,
                "sample_count": recording.snapshots.len(),
                "camera_mode": if self.controls.mounted { "mounted" } else { "orbit" },
                "environment_id": recording.header.environment_ids[self.environment_index],
                "snapshot": snapshot,
            });
            info["camera"] = serde_json::json!({
                "near": recording.header.camera.near,
                "far": recording.header.camera.far,
                "fov_vertical_degrees": recording.header.camera.vertical_fov_radians.to_degrees(),
                "width": self.sensor_size[0], "height": self.sensor_size[1],
            });
            let mut objects = vec![serde_json::json!({"id": GROUND_OBJECT_ID, "name": "ground"})];
            for vehicle in &snapshot.vehicles {
                objects.push(serde_json::json!({
                    "id": vehicle.drone_object_id(), "name": "drone",
                    "environment_id": vehicle.environment_id,
                }));
                objects.push(serde_json::json!({
                    "id": vehicle.target_object_id(), "name": "target",
                    "environment_id": vehicle.environment_id,
                }));
            }
            info["objects"] = objects.into();
        }
        info.to_string()
    }
}
