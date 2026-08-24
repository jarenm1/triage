use std::{sync::Arc, time::Instant};

use sim_graphics::{DepthTarget, Frame, RenderTarget, Renderer, RendererError};
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalSize, PhysicalSize},
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

pub trait Scene {
    fn initialize(&mut self, _renderer: &mut Renderer) -> Result<(), RendererError> {
        Ok(())
    }

    fn frame(&mut self, elapsed_seconds: f32) -> &Frame;
}

#[derive(Debug)]
pub enum WindowError {
    EventLoop(winit::error::EventLoopError),
    Window(winit::error::OsError),
    CreateSurface(wgpu::CreateSurfaceError),
    Renderer(RendererError),
    SurfaceValidation,
}

impl std::fmt::Display for WindowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EventLoop(error) => write!(f, "window event loop failed: {error}"),
            Self::Window(error) => write!(f, "creating the window failed: {error}"),
            Self::CreateSurface(error) => write!(f, "creating the window surface failed: {error}"),
            Self::Renderer(error) => write!(f, "creating the renderer failed: {error}"),
            Self::SurfaceValidation => write!(f, "surface texture acquisition failed validation"),
        }
    }
}

impl std::error::Error for WindowError {}

impl From<winit::error::EventLoopError> for WindowError {
    fn from(error: winit::error::EventLoopError) -> Self {
        Self::EventLoop(error)
    }
}

pub fn run(title: impl Into<String>, scene: impl Scene + 'static) -> Result<(), WindowError> {
    let event_loop = EventLoop::new()?;
    let mut application = WindowApplication {
        title: title.into(),
        scene,
        started: Instant::now(),
        window: None,
        error: None,
    };
    event_loop.run_app(&mut application)?;
    match application.error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

struct WindowApplication<S> {
    title: String,
    scene: S,
    started: Instant,
    window: Option<WindowState>,
    error: Option<WindowError>,
}

struct WindowState {
    instance: wgpu::Instance,
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    renderer: Renderer,
    depth: DepthTarget,
    config: wgpu::SurfaceConfiguration,
}

impl<S: Scene> ApplicationHandler for WindowApplication<S> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() || self.error.is_some() {
            return;
        }
        match pollster::block_on(WindowState::new(event_loop, &self.title)) {
            Ok(mut window) => {
                if let Err(error) = self.scene.initialize(&mut window.renderer) {
                    self.error = Some(WindowError::Renderer(error));
                    event_loop.exit();
                    return;
                }
                self.window = Some(window);
                self.started = Instant::now();
            }
            Err(error) => {
                self.error = Some(error);
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.as_mut() else {
            return;
        };
        if window.window.id() != window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => window.resize(size),
            WindowEvent::RedrawRequested => {
                profiling::scope!("window frame");
                let frame = self.scene.frame(self.started.elapsed().as_secs_f32());
                if let Err(error) = window.render(frame) {
                    self.error = Some(error);
                    event_loop.exit();
                }
                profiling::finish_frame!();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.window.request_redraw();
        }
    }
}

impl WindowState {
    async fn new(event_loop: &ActiveEventLoop, title: &str) -> Result<Self, WindowError> {
        let attributes = WindowAttributes::default()
            .with_title(title)
            .with_inner_size(LogicalSize::new(1280, 720));
        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .map_err(WindowError::Window)?,
        );
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle(
            Box::new(window.clone()),
        ));
        let surface = instance
            .create_surface(window.clone())
            .map_err(WindowError::CreateSurface)?;
        let renderer = Renderer::new(&instance, Some(&surface))
            .await
            .map_err(WindowError::Renderer)?;
        let capabilities = surface.get_capabilities(renderer.adapter());
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: renderer.color_format(),
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoNoVsync,
            desired_maximum_frame_latency: 1,
            alpha_mode: capabilities.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(renderer.device(), &config);
        let depth = DepthTarget::new(&renderer, config.width, config.height);
        Ok(Self {
            window,
            instance,
            surface,
            renderer,
            depth,
            config,
        })
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(self.renderer.device(), &self.config);
        self.depth = DepthTarget::new(&self.renderer, size.width, size.height);
    }

    fn render(&mut self, frame: &Frame) -> Result<(), WindowError> {
        profiling::function_scope!();
        let (surface_texture, reconfigure_after_present) = match self.surface.get_current_texture()
        {
            wgpu::CurrentSurfaceTexture::Success(texture) => (texture, false),
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => (texture, true),
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.resize(self.window.inner_size());
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                self.surface = self
                    .instance
                    .create_surface(self.window.clone())
                    .map_err(WindowError::CreateSurface)?;
                self.resize(self.window.inner_size());
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(WindowError::SurfaceValidation);
            }
        };
        let color = surface_texture.texture.create_view(&Default::default());
        self.renderer
            .render(
                RenderTarget {
                    color: &color,
                    depth: self.depth.view(),
                    width: self.config.width,
                    height: self.config.height,
                },
                frame,
            )
            .map_err(WindowError::Renderer)?;
        self.renderer.present(surface_texture);
        if reconfigure_after_present {
            self.resize(self.window.inner_size());
        }
        Ok(())
    }
}
