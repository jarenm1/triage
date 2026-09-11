use std::{path::PathBuf, sync::Arc};

use anyhow::Result;
use glam::{Quat, Vec3};
use sim_graphics_winit::winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};
use sim_inspection::GpuInspector;
use sim_scene::SceneConfig;

pub fn run(path: Option<PathBuf>) -> Result<()> {
    let path = path.unwrap_or_else(|| PathBuf::from("target/inspection-scene.json"));
    let (config, status) = if path.exists() {
        match SceneConfig::load(&path).and_then(|config| {
            config.validate()?;
            Ok(config)
        }) {
            Ok(config) => (config, format!("Loaded {}", path.display())),
            Err(error) => (
                SceneConfig::default(),
                format!("Load failed; showing default: {error:#}"),
            ),
        }
    } else {
        (
            SceneConfig::default(),
            format!("Default scene; S saves to {}", path.display()),
        )
    };
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = InspectionApp {
        window: None,
        instance: None,
        surface: None,
        inspector: None,
        surface_config: None,
        config,
        path,
        status,
        fatal: None,
    };
    event_loop.run_app(&mut app)?;
    if let Some(error) = app.fatal {
        anyhow::bail!(error);
    }
    Ok(())
}

struct InspectionApp {
    window: Option<Arc<Window>>,
    instance: Option<wgpu::Instance>,
    surface: Option<wgpu::Surface<'static>>,
    inspector: Option<GpuInspector>,
    surface_config: Option<wgpu::SurfaceConfiguration>,
    config: SceneConfig,
    path: PathBuf,
    status: String,
    fatal: Option<String>,
}

impl InspectionApp {
    fn update_scene(&mut self, candidate: SceneConfig, success: &str) {
        if let Err(error) = candidate.validate() {
            self.status = format!("Invalid scene (unchanged): {error:#}");
            return;
        }
        self.config = candidate;
        self.status = success.to_owned();
    }

    fn key(&mut self, key: KeyCode, repeat: bool, event_loop: &ActiveEventLoop) {
        if key == KeyCode::Escape {
            event_loop.exit();
            return;
        }
        let mut next = self.config.clone();
        let eye = Vec3::from_array(next.sensor.eye);
        let target = Vec3::from_array(next.sensor.target);
        let offset = eye - target;
        match key {
            KeyCode::ArrowLeft | KeyCode::ArrowRight => {
                let angle = if key == KeyCode::ArrowLeft {
                    0.08
                } else {
                    -0.08
                };
                next.sensor.eye = (target
                    + Quat::from_axis_angle(Vec3::from_array(next.sensor.up).normalize(), angle)
                        * offset)
                    .to_array();
            }
            KeyCode::ArrowUp | KeyCode::ArrowDown => {
                let angle = if key == KeyCode::ArrowUp { 0.06 } else { -0.06 };
                let right = offset
                    .cross(Vec3::from_array(next.sensor.up))
                    .normalize_or_zero();
                if right.length_squared() < 0.5 {
                    return;
                }
                let rotated = Quat::from_axis_angle(right, angle) * offset;
                if rotated
                    .normalize_or_zero()
                    .dot(Vec3::from_array(next.sensor.up).normalize_or_zero())
                    .abs()
                    > 0.98
                {
                    return;
                }
                next.sensor.eye = (target + rotated).to_array();
            }
            KeyCode::Equal | KeyCode::NumpadAdd | KeyCode::Minus | KeyCode::NumpadSubtract => {
                let factor = if matches!(key, KeyCode::Equal | KeyCode::NumpadAdd) {
                    0.9
                } else {
                    1.1
                };
                let distance = (offset.length() * factor).clamp(
                    next.sensor.near * 2.0,
                    next.sensor.far.max(next.sensor.near * 2.0),
                );
                next.sensor.eye = (target + offset.normalize_or_zero() * distance).to_array();
            }
            KeyCode::PageUp | KeyCode::PageDown => {
                let delta = Vec3::from_array(next.sensor.up).normalize()
                    * if key == KeyCode::PageUp { 0.25 } else { -0.25 };
                next.sensor.eye = (eye + delta).to_array();
                next.sensor.target = (target + delta).to_array();
            }
            KeyCode::BracketLeft | KeyCode::BracketRight => {
                let delta = if key == KeyCode::BracketLeft {
                    -0.04
                } else {
                    0.04
                };
                next.sensor.vertical_fov_radians =
                    (next.sensor.vertical_fov_radians + delta).clamp(0.15, 2.8);
            }
            KeyCode::KeyS if !repeat => {
                let result = (|| {
                    if let Some(parent) = self.path.parent().filter(|p| !p.as_os_str().is_empty()) {
                        std::fs::create_dir_all(parent)?;
                    }
                    self.config.save(&self.path)
                })();
                self.status = match result {
                    Ok(()) => format!("Saved scene: {}", self.path.display()),
                    Err(error) => format!("Save failed: {error:#}"),
                };
                return;
            }
            KeyCode::KeyL if !repeat => {
                match SceneConfig::load(&self.path) {
                    Ok(config) => self.update_scene(config, "Reloaded scene"),
                    Err(error) => {
                        self.status = format!("Reload failed (scene unchanged): {error:#}")
                    }
                }
                return;
            }
            KeyCode::KeyR if !repeat => {
                self.update_scene(SceneConfig::default(), "Reset to default scene (not saved)");
                return;
            }
            KeyCode::KeyE if !repeat => {
                self.status = match self.inspector.as_mut() {
                    Some(inspector) => {
                        let result = (|| -> Result<PathBuf> {
                            let capture = inspector.capture(&self.config)?;
                            let timestamp = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)?
                                .as_nanos();
                            let root = PathBuf::from("target/inspection");
                            std::fs::create_dir_all(&root)?;
                            let mut sequence = 0_u64;
                            let directory = loop {
                                let directory = root.join(format!(
                                    "capture-{timestamp}-{}-{sequence}",
                                    std::process::id()
                                ));
                                match std::fs::create_dir(&directory) {
                                    Ok(()) => break directory,
                                    Err(error)
                                        if error.kind() == std::io::ErrorKind::AlreadyExists =>
                                    {
                                        sequence += 1;
                                    }
                                    Err(error) => return Err(error.into()),
                                }
                            };
                            capture.save(&self.config, &directory)?;
                            Ok(directory)
                        })();
                        match result {
                            Ok(directory) => {
                                format!("Explicit export saved: {}", directory.display())
                            }
                            Err(error) => format!("Explicit export failed: {error:#}"),
                        }
                    }
                    None => "Export unavailable: GPU renderer is not initialized".to_owned(),
                };
                return;
            }
            KeyCode::Space if !repeat => return,
            _ => return,
        }
        self.update_scene(
            next,
            "Scene updated | S saves scene | E explicitly exports sensors",
        );
    }

    fn draw(&mut self) -> Result<()> {
        let Some(window) = self.window.as_ref() else {
            return Ok(());
        };
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }
        let (Some(surface), Some(inspector), Some(config)) = (
            self.surface.as_mut(),
            self.inspector.as_mut(),
            self.surface_config.as_mut(),
        ) else {
            return Ok(());
        };
        if config.width != size.width || config.height != size.height {
            config.width = size.width;
            config.height = size.height;
            surface.configure(inspector.device(), config);
        }
        let (frame, suboptimal) = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => (frame, false),
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => (frame, true),
            wgpu::CurrentSurfaceTexture::Timeout => {
                self.status = "Surface timed out; retrying presentation".to_owned();
                window.set_title(&self.status);
                window.request_redraw();
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Occluded => return Ok(()),
            wgpu::CurrentSurfaceTexture::Outdated => {
                surface.configure(inspector.device(), config);
                window.request_redraw();
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                let instance = self
                    .instance
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("GPU instance is unavailable"))?;
                *surface = instance.create_surface(window.clone())?;
                surface.configure(inspector.device(), config);
                window.request_redraw();
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                anyhow::bail!("GPU surface validation failed");
            }
        };
        let sensor = &self.config.sensor;
        let status = [
            "Arrows: orbit | +/-: dolly | PgUp/PgDn: height | [/]: FOV".to_owned(),
            "S: save | L: reload | R: reset | Space: redraw | Esc: close".to_owned(),
            "E: explicit synchronous sensor export (may pause; no readback during navigation)"
                .to_owned(),
            format!(
                "Eye [{:.2}, {:.2}, {:.2}] Target [{:.2}, {:.2}, {:.2}]",
                sensor.eye[0],
                sensor.eye[1],
                sensor.eye[2],
                sensor.target[0],
                sensor.target[1],
                sensor.target[2],
            ),
            format!(
                "Sensor {}x{} | vertical FOV {:.1} deg | clip {:.3}..{:.1}",
                sensor.width,
                sensor.height,
                sensor.vertical_fov_radians.to_degrees(),
                sensor.near,
                sensor.far,
            ),
            format!("Scene: {}", self.path.display()),
            self.status.clone(),
        ];
        let view = frame.texture.create_view(&Default::default());
        inspector.render(&self.config, &view, size.width, size.height, &status)?;
        window.pre_present_notify();
        inspector.queue().present(frame);
        if suboptimal {
            surface.configure(inspector.device(), config);
        }
        window.set_title("Sensor Inspector - RGB / Depth / Instances / Overview");
        Ok(())
    }
}

impl ApplicationHandler for InspectionApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let result = (|| -> Result<()> {
            let window = Arc::new(
                event_loop.create_window(
                    Window::default_attributes()
                        .with_title("Sensor Inspector - RGB / Depth / Instances / Overview")
                        .with_inner_size(LogicalSize::new(1280.0, 1000.0)),
                )?,
            );
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle(
                Box::new(window.clone()),
            ));
            let surface = instance.create_surface(window.clone())?;
            let inspector = pollster::block_on(GpuInspector::new(&instance, Some(&surface)))?;
            let size = window.inner_size();
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: inspector.format(),
                color_space: wgpu::SurfaceColorSpace::Auto,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::AutoVsync,
                desired_maximum_frame_latency: 1,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
            };
            if size.width > 0 && size.height > 0 {
                surface.configure(inspector.device(), &config);
            }
            self.window = Some(window);
            self.instance = Some(instance);
            self.surface = Some(surface);
            self.inspector = Some(inspector);
            self.surface_config = Some(config);
            Ok(())
        })();
        if let Err(error) = result {
            self.fatal = Some(format!("Unable to create inspection window: {error:#}"));
            event_loop.exit();
            return;
        }
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if self
            .window
            .as_ref()
            .is_none_or(|window| window.id() != window_id)
        {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_) | WindowEvent::Occluded(false) => {
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    self.key(key, event.repeat, event_loop);
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let Err(error) = self.draw() {
                    self.status = format!("Presentation failed: {error:#}");
                    if let Some(window) = &self.window {
                        window.set_title(&self.status);
                    }
                    eprintln!("{}", self.status);
                }
            }
            _ => {}
        }
    }
}
