use std::sync::Arc;

use sim_graphics::{
    DepthTarget, ExternalView, Frame, OutputKind, ReadbackData, ReadbackHandle, Renderer,
    RendererError, ViewKey, ViewKind,
};
use web_time::Instant;
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalSize, PhysicalSize},
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    window::{Window, WindowAttributes, WindowId},
};
pub use winit;


pub trait Scene {
    fn initialize(&mut self, _renderer: &mut Renderer) -> Result<(), RendererError> {
        Ok(())
    }

    fn frame(&mut self, elapsed_seconds: f32, width: u32, height: u32) -> &Frame;

    fn sensor_output(&mut self, _view: ViewKey, _output: OutputKind, _data: ReadbackData) {}

    fn window_event(&mut self, _event: &WindowEvent) {}
}

#[derive(Debug, Clone)]
pub struct WindowConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub canvas_id: Option<String>,
    pub append_to_body: bool,
}

impl WindowConfig {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            width: 1280,
            height: 720,
            canvas_id: None,
            append_to_body: true,
        }
    }

    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn with_canvas_id(mut self, canvas_id: impl Into<String>) -> Self {
        self.canvas_id = Some(canvas_id.into());
        self
    }

    pub fn with_append_to_body(mut self, append: bool) -> Self {
        self.append_to_body = append;
        self
    }
}

#[derive(Debug)]
pub enum WindowError {
    EventLoop(winit::error::EventLoopError),
    Window(winit::error::OsError),
    CreateSurface(wgpu::CreateSurfaceError),
    Renderer(RendererError),
    MissingDisplayView,
    SurfaceValidation,
}

impl std::fmt::Display for WindowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EventLoop(error) => write!(f, "window event loop failed: {error}"),
            Self::Window(error) => write!(f, "creating the window failed: {error}"),
            Self::CreateSurface(error) => write!(f, "creating the window surface failed: {error}"),
            Self::Renderer(error) => write!(f, "creating the renderer failed: {error}"),
            Self::MissingDisplayView => write!(f, "window frame has no display view"),
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

pub enum AppEvent {
    Initialized(WindowState),
    InitFailed(WindowError),
}

pub fn run(title: impl Into<String>, scene: impl Scene + 'static) -> Result<(), WindowError> {
    run_with_config(WindowConfig::new(title), scene)
}

pub fn run_with_config(
    config: WindowConfig,
    scene: impl Scene + 'static,
) -> Result<(), WindowError> {
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Info).ok();
    }

    let event_loop = EventLoop::<AppEvent>::with_user_event().build()?;
    let proxy = event_loop.create_proxy();
    #[allow(unused_mut)]
    let mut application = WindowApplication {
        proxy,
        state: AppState::Uninitialized { config, scene },
        error: None,
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        event_loop.run_app(&mut application)?;
        match application.error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn_app(application);
        Ok(())
    }
}

enum AppState<S> {
    Uninitialized {
        config: WindowConfig,
        scene: S,
    },
    #[allow(dead_code)]
    Initializing {
        scene: Option<S>,
    },
    Running {
        window: WindowState,
        scene: S,
        started: Instant,
    },
    Failed,
}

struct WindowApplication<S> {
    #[allow(dead_code)]
    proxy: EventLoopProxy<AppEvent>,
    state: AppState<S>,
    error: Option<WindowError>,
}

pub struct WindowState {
    instance: wgpu::Instance,
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    renderer: Renderer,
    depth: DepthTarget,
    config: wgpu::SurfaceConfiguration,
    pending_readbacks: Vec<ReadbackHandle>,
}

#[cfg(target_arch = "wasm32")]
fn find_canvas(config: &WindowConfig) -> Option<web_sys::HtmlCanvasElement> {
    use wasm_bindgen::JsCast;
    let window = web_sys::window()?;
    let document = window.document()?;
    if let Some(id) = &config.canvas_id {
        return document
            .get_element_by_id(id)
            .and_then(|el| el.dyn_into::<web_sys::HtmlCanvasElement>().ok());
    }
    if let Some(el) = document.get_element_by_id("sim-canvas") {
        if let Ok(canvas) = el.dyn_into::<web_sys::HtmlCanvasElement>() {
            return Some(canvas);
        }
    }
    if let Some(el) = document.get_element_by_id("canvas") {
        if let Ok(canvas) = el.dyn_into::<web_sys::HtmlCanvasElement>() {
            return Some(canvas);
        }
    }
    None
}

impl<S: Scene + 'static> ApplicationHandler<AppEvent> for WindowApplication<S> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let AppState::Uninitialized { config, scene } =
            std::mem::replace(&mut self.state, AppState::Failed)
        else {
            return;
        };

        #[allow(unused_mut)]
        let mut attributes = WindowAttributes::default()
            .with_title(&config.title)
            .with_inner_size(LogicalSize::new(config.width, config.height));

        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowAttributesExtWebSys;
            let canvas = find_canvas(&config);
            if let Some(canvas) = canvas {
                attributes = attributes.with_canvas(Some(canvas));
            } else if config.append_to_body {
                attributes = attributes.with_append(true);
            }
        }

        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                self.error = Some(WindowError::Window(error));
                event_loop.exit();
                return;
            }
        };

        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut window_state = match pollster::block_on(WindowState::new(window)) {
                Ok(state) => state,
                Err(error) => {
                    self.error = Some(error);
                    event_loop.exit();
                    return;
                }
            };
            let mut scene = scene;
            if let Err(error) = scene.initialize(&mut window_state.renderer) {
                self.error = Some(WindowError::Renderer(error));
                event_loop.exit();
                return;
            }
            self.state = AppState::Running {
                window: window_state,
                scene,
                started: Instant::now(),
            };
        }

        #[cfg(target_arch = "wasm32")]
        {
            let proxy = self.proxy.clone();
            self.state = AppState::Initializing { scene: Some(scene) };
            wasm_bindgen_futures::spawn_local(async move {
                match WindowState::new(window).await {
                    Ok(state) => {
                        let _ = proxy.send_event(AppEvent::Initialized(state));
                    }
                    Err(error) => {
                        let _ = proxy.send_event(AppEvent::InitFailed(error));
                    }
                }
            });
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: AppEvent) {
        match event {
            AppEvent::Initialized(mut window_state) => {
                if let AppState::Initializing {
                    scene: Some(mut scene),
                } = std::mem::replace(&mut self.state, AppState::Failed)
                {
                    if let Err(error) = scene.initialize(&mut window_state.renderer) {
                        self.error = Some(WindowError::Renderer(error));
                        event_loop.exit();
                        return;
                    }
                    window_state.window.request_redraw();
                    self.state = AppState::Running {
                        window: window_state,
                        scene,
                        started: Instant::now(),
                    };
                }
            }
            AppEvent::InitFailed(error) => {
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
        let AppState::Running {
            window,
            scene,
            started,
        } = &mut self.state
        else {
            return;
        };

        if window.window.id() != window_id {
            return;
        }

        scene.window_event(&event);

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => window.resize(size),
            WindowEvent::RedrawRequested => {
                profiling::scope!("window frame");
                let frame = scene.frame(
                    started.elapsed().as_secs_f32(),
                    window.config.width,
                    window.config.height,
                );
                if let Err(error) = window.render(frame) {
                    self.error = Some(error);
                    event_loop.exit();
                }
                if let Err(error) = window.poll_outputs(scene) {
                    self.error = Some(error);
                    event_loop.exit();
                }
                window.window.request_redraw();
                profiling::finish_frame!();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let AppState::Running { window, .. } = &self.state {
            window.window.request_redraw();
        }
    }
}

impl WindowState {
    async fn new(window: Arc<Window>) -> Result<Self, WindowError> {
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
        let present_mode = if cfg!(target_arch = "wasm32") {
            capabilities
                .present_modes
                .first()
                .copied()
                .unwrap_or(wgpu::PresentMode::Fifo)
        } else if capabilities
            .present_modes
            .contains(&wgpu::PresentMode::AutoNoVsync)
        {
            wgpu::PresentMode::AutoNoVsync
        } else {
            capabilities
                .present_modes
                .first()
                .copied()
                .unwrap_or(wgpu::PresentMode::Fifo)
        };
        let alpha_mode = capabilities
            .alpha_modes
            .first()
            .copied()
            .unwrap_or(wgpu::CompositeAlphaMode::Auto);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: renderer.color_format(),
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            desired_maximum_frame_latency: 1,
            alpha_mode,
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
            pending_readbacks: Vec::new(),
        })
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn renderer(&self) -> &Renderer {
        &self.renderer
    }

    pub fn renderer_mut(&mut self) -> &mut Renderer {
        &mut self.renderer
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(self.renderer.device(), &self.config);
        self.depth = DepthTarget::new(&self.renderer, size.width, size.height);
    }

    pub fn render(&mut self, frame: &Frame) -> Result<(), WindowError> {
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
        let display_view = frame
            .first_view(ViewKind::Display)
            .ok_or(WindowError::MissingDisplayView)?;
        let submission = self
            .renderer
            .execute(
                frame,
                &[ExternalView {
                    view: display_view,
                    color: &color,
                    depth: self.depth.view(),
                }],
            )
            .map_err(WindowError::Renderer)?;
        self.pending_readbacks.extend(submission.readbacks);
        self.renderer.present(surface_texture);
        if reconfigure_after_present {
            self.resize(self.window.inner_size());
        }
        Ok(())
    }

    pub fn poll_outputs(&mut self, scene: &mut impl Scene) -> Result<(), WindowError> {
        profiling::scope!("poll sensor readbacks");
        let mut index = 0;
        while index < self.pending_readbacks.len() {
            let handle = self.pending_readbacks[index];
            match self
                .renderer
                .poll_readback(handle)
                .map_err(WindowError::Renderer)?
            {
                Some(data) => {
                    self.pending_readbacks.swap_remove(index);
                    scene.sensor_output(handle.view(), handle.output(), data);
                }
                None => index += 1,
            }
        }
        Ok(())
    }
}
