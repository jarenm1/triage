use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

mod camera;
mod frame;
mod graph;
mod primitive;
mod resource;

pub use camera::Camera;
pub use frame::{Frame, RenderView, ViewId, ViewKey, ViewKind, ViewOutputs};
pub use graph::{ExternalView, FrameSubmission, OutputKind, ReadbackData, ReadbackHandle};
pub use primitive::{Mesh, MeshData, MeshHandle, MeshVertex, RenderPrimitive};
pub use resource::{Handle, ResourceRegistry};

const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[derive(Debug)]
pub enum RendererError {
    NoAdapter(wgpu::RequestAdapterError),
    RequestDevice(wgpu::RequestDeviceError),
    TooManyInstances(usize),
    TooManyIndices(usize),
    EmptyMesh,
    InvalidMeshHandle(MeshHandle),
    ReadbackFailed,
    ReadbackRingFull { view: ViewKey, output: OutputKind },
    MissingExternalView(ViewId),
    InvalidReadbackHandle,
    SkyboxImageError(String),
}

impl std::fmt::Display for RendererError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAdapter(error) => write!(f, "no compatible graphics adapter found: {error}"),
            Self::RequestDevice(error) => write!(f, "requesting a graphics device failed: {error}"),
            Self::TooManyInstances(count) => {
                write!(f, "{count} instances exceed the GPU draw limit")
            }
            Self::TooManyIndices(count) => {
                write!(f, "{count} mesh indices exceed the GPU draw limit")
            }
            Self::EmptyMesh => write!(f, "meshes require at least one vertex and one index"),
            Self::InvalidMeshHandle(handle) => {
                write!(f, "frame references missing mesh {handle:?}")
            }
            Self::ReadbackFailed => write!(f, "GPU readback failed"),
            Self::ReadbackRingFull { view, output } => {
                write!(
                    f,
                    "readback ring for view {view:?} output {output:?} is full"
                )
            }
            Self::InvalidReadbackHandle => write!(f, "readback handle is stale or failed"),
            Self::MissingExternalView(view) => {
                write!(f, "display view {view:?} has no external target")
            }
            Self::SkyboxImageError(error) => {
                write!(f, "failed to load skybox image: {error}")
            }
        }
    }
}

impl std::error::Error for RendererError {}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Instance {
    model: [[f32; 4]; 4],
    color: [f32; 4],
    object_id: u32,
    padding: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    view: [[f32; 4]; 4],
    view_projection: [[f32; 4]; 4],
    inv_view_projection: [[f32; 4]; 4],
    camera_position: [f32; 4],
    light_direction: [f32; 4],
    light_color: [f32; 4],
    ambient_color: [f32; 4],
}

#[derive(Clone, Copy)]
struct DrawBatch {
    mesh: MeshHandle,
    first_instance: u32,
    instance_count: u32,
}

#[derive(Clone, Copy)]
struct CompiledView {
    request: RenderView,
    external: Option<usize>,
    color: Option<usize>,
    depth: Option<usize>,
    object_id: Option<usize>,
    depth_attachment: Option<usize>,
}
pub struct Renderer {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    color_format: wgpu::TextureFormat,
    display_pipeline: wgpu::RenderPipeline,
    sensor_pipelines: Vec<Option<wgpu::RenderPipeline>>,
    skybox_pipeline: wgpu::RenderPipeline,
    skybox_bind_group_layout: wgpu::BindGroupLayout,
    skybox_texture: wgpu::Texture,
    skybox_bind_group: wgpu::BindGroup,
    globals: wgpu::Buffer,
    globals_layout: wgpu::BindGroupLayout,
    globals_capacity: usize,
    globals_stride: u32,
    globals_bind_group: wgpu::BindGroup,
    meshes: ResourceRegistry<Mesh>,
    instance_buffer: wgpu::Buffer,
    instance_capacity: usize,
    instances: Vec<Instance>,
    batches: Vec<DrawBatch>,
    texture_pool: graph::TexturePool,
    readbacks: graph::ReadbackRing,
    compiled_views: Vec<CompiledView>,
}

pub struct RenderTarget<'a> {
    pub color: &'a wgpu::TextureView,
    pub depth: &'a wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

impl Renderer {
    pub async fn new(
        instance: &wgpu::Instance,
        compatible_surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Self, RendererError> {
        #[cfg(not(target_arch = "wasm32"))]
        let _profiler = profiling::tracy_client::Client::start();
        profiling::function_scope!();
        let preferred_adapter = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        };
        let adapter = match instance.request_adapter(&preferred_adapter).await {
            Ok(adapter) => adapter,
            Err(_) => instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    force_fallback_adapter: true,
                    ..preferred_adapter
                })
                .await
                .map_err(RendererError::NoAdapter)?,
        };
        let color_format = compatible_surface
            .and_then(|surface| {
                let capabilities = surface.get_capabilities(&adapter);
                capabilities
                    .formats
                    .iter()
                    .copied()
                    .find(wgpu::TextureFormat::is_srgb)
                    .or_else(|| capabilities.formats.first().copied())
            })
            .unwrap_or(COLOR_FORMAT);
        let limits = if cfg!(target_arch = "wasm32") {
            wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits())
        } else {
            wgpu::Limits::default()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("sim-graphics device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(RendererError::RequestDevice)?;

        let globals_capacity = 8;
        let globals_stride = device
            .limits()
            .min_uniform_buffer_offset_alignment
            .max(u32::try_from(size_of::<Globals>()).expect("globals size exceeds u32"));
        let globals = create_globals_buffer(&device, globals_capacity, globals_stride);
        let globals_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene globals layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(size_of::<Globals>() as u64),
                },
                count: None,
            }],
        });
        let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene globals bind group"),
            layout: &globals_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &globals,
                    offset: 0,
                    size: wgpu::BufferSize::new(size_of::<Globals>() as u64),
                }),
            }],
        });
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("primitive pipeline layout"),
            bind_group_layouts: &[Some(&globals_layout)],
            immediate_size: 0,
        });
        let display_pipeline = create_render_pipeline(
            &device,
            &pipeline_layout,
            &shader,
            "display primitive pipeline",
            "fs_display",
            &[Some(color_target(color_format))],
        );
        let mut sensor_pipelines = Vec::with_capacity(8);
        sensor_pipelines.push(None);
        for mask in 1_u8..8 {
            let targets = [
                (mask & ViewOutputs::COLOR.bits() != 0)
                    .then(|| color_target(graph::SENSOR_COLOR_FORMAT)),
                (mask & ViewOutputs::DEPTH.bits() != 0)
                    .then(|| color_target(graph::SENSOR_DEPTH_FORMAT)),
                (mask & ViewOutputs::OBJECT_ID.bits() != 0)
                    .then(|| color_target(graph::SENSOR_OBJECT_ID_FORMAT)),
            ];
            sensor_pipelines.push(Some(create_render_pipeline(
                &device,
                &pipeline_layout,
                &shader,
                "sensor primitive pipeline",
                "fs_sensor",
                &targets,
            )));
        }
        let skybox_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("skybox bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let default_sky_width = 512;
        let default_sky_height = 256;
        let default_sky_data = generate_default_skybox_rgba(default_sky_width, default_sky_height);
        let (skybox_texture, skybox_bind_group) = create_skybox_texture_and_bind_group(
            &device,
            &queue,
            &skybox_bind_group_layout,
            default_sky_width,
            default_sky_height,
            &default_sky_data,
        );
        let skybox_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("skybox pipeline layout"),
                bind_group_layouts: &[Some(&globals_layout), Some(&skybox_bind_group_layout)],
                immediate_size: 0,
            });
        let skybox_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("skybox pipeline"),
            layout: Some(&skybox_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_skybox"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_skybox"),
                compilation_options: Default::default(),
                targets: &[Some(color_target(color_format))],
            }),
            multiview_mask: None,
            cache: None,
        });
        let instance_capacity = 64;
        let instance_buffer = create_instance_buffer(&device, instance_capacity);
        Ok(Self {
            adapter,
            device,
            queue,
            color_format,
            display_pipeline,
            sensor_pipelines,
            skybox_pipeline,
            skybox_bind_group_layout,
            skybox_texture,
            skybox_bind_group,
            globals,
            globals_layout,
            globals_capacity,
            globals_stride,
            globals_bind_group,
            meshes: ResourceRegistry::new(),
            instance_buffer,
            instance_capacity,
            instances: Vec::with_capacity(instance_capacity),
            batches: Vec::new(),
            texture_pool: graph::TexturePool::new(),
            readbacks: graph::ReadbackRing::new(),
            compiled_views: Vec::new(),
        })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }

    pub fn color_format(&self) -> wgpu::TextureFormat {
        self.color_format
    }

    pub fn present(&self, surface_texture: wgpu::SurfaceTexture) {
        self.queue.present(surface_texture);
    }

    pub fn register_mesh(&mut self, data: MeshData) -> Result<MeshHandle, RendererError> {
        profiling::function_scope!();
        if data.vertices.is_empty() || data.indices.is_empty() {
            return Err(RendererError::EmptyMesh);
        }
        let index_count = u32::try_from(data.indices.len())
            .map_err(|_| RendererError::TooManyIndices(data.indices.len()))?;
        let vertices = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh vertices"),
                contents: bytemuck::cast_slice(&data.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let indices = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh indices"),
                contents: bytemuck::cast_slice(&data.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        Ok(self.meshes.insert(Mesh {
            vertices,
            indices,
            index_count,
        }))
    }

    pub fn set_skybox_rgba(
        &mut self,
        width: u32,
        height: u32,
        rgba_data: &[u8],
    ) -> Result<(), RendererError> {
        if (width * height * 4) as usize != rgba_data.len() {
            return Err(RendererError::SkyboxImageError(format!(
                "RGBA data length {} does not match dimensions {}x{}",
                rgba_data.len(),
                width,
                height
            )));
        }
        let (texture, bind_group) = create_skybox_texture_and_bind_group(
            &self.device,
            &self.queue,
            &self.skybox_bind_group_layout,
            width,
            height,
            rgba_data,
        );
        self.skybox_texture = texture;
        self.skybox_bind_group = bind_group;
        Ok(())
    }

    pub fn set_skybox_from_image_bytes(&mut self, bytes: &[u8]) -> Result<(), RendererError> {
        let img = image::load_from_memory(bytes)
            .map_err(|err| RendererError::SkyboxImageError(err.to_string()))?;
        let rgba = dynamic_image_to_srgb_rgba8(&img);
        self.set_skybox_rgba(rgba.width(), rgba.height(), &rgba)
    }

    pub fn load_skybox(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), RendererError> {
        let img = image::open(path)
            .map_err(|err| RendererError::SkyboxImageError(err.to_string()))?;
        let rgba = dynamic_image_to_srgb_rgba8(&img);
        self.set_skybox_rgba(rgba.width(), rgba.height(), &rgba)
    }

    pub fn remove_mesh(&mut self, handle: MeshHandle) -> bool {
        self.meshes.remove(handle).is_some()
    }

    pub fn execute(
        &mut self,
        frame: &Frame,
        external_views: &[ExternalView<'_>],
    ) -> Result<FrameSubmission, RendererError> {
        profiling::function_scope!();
        {
            profiling::scope!("extract world");
            let primitive_count = frame.primitives().len();
            u32::try_from(primitive_count)
                .map_err(|_| RendererError::TooManyInstances(primitive_count))?;
            self.ensure_instance_capacity(primitive_count);
            self.ensure_globals_capacity(frame.views().len());
            self.instances.clear();
            self.batches.clear();
            for primitive in frame.primitives() {
                if self.meshes.get(primitive.mesh).is_none() {
                    return Err(RendererError::InvalidMeshHandle(primitive.mesh));
                }
                self.instances.push(Instance {
                    model: primitive.transform.to_cols_array_2d(),
                    color: primitive.color,
                    object_id: primitive.object_id,
                    padding: [0; 3],
                });
                match self.batches.last_mut() {
                    Some(batch) if batch.mesh == primitive.mesh => batch.instance_count += 1,
                    _ => self.batches.push(DrawBatch {
                        mesh: primitive.mesh,
                        first_instance: (self.instances.len() - 1) as u32,
                        instance_count: 1,
                    }),
                }
            }
        }

        {
            profiling::scope!("compile frame graph");
            self.texture_pool.begin_frame();
            self.compiled_views.clear();
            for (view_index, view) in frame.views().iter().copied().enumerate() {
                match view.kind {
                    ViewKind::Display => {
                        let external = external_views
                            .iter()
                            .position(|external| external.view.index() == view_index)
                            .ok_or(RendererError::MissingExternalView(ViewId::from_index(
                                view_index,
                            )))?;
                        self.compiled_views.push(CompiledView {
                            request: view,
                            external: Some(external),
                            color: None,
                            depth: None,
                            object_id: None,
                            depth_attachment: None,
                        });
                    }
                    ViewKind::Sensor => {
                        profiling::scope!("allocate transient resources");
                        let usage =
                            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC;
                        let mut acquire = |format| {
                            self.texture_pool.acquire(
                                &self.device,
                                graph::TextureKey {
                                    width: view.width,
                                    height: view.height,
                                    format,
                                    usage,
                                },
                            )
                        };
                        let color = view
                            .outputs
                            .contains(ViewOutputs::COLOR)
                            .then(|| acquire(graph::SENSOR_COLOR_FORMAT));
                        let depth = view
                            .outputs
                            .contains(ViewOutputs::DEPTH)
                            .then(|| acquire(graph::SENSOR_DEPTH_FORMAT));
                        let object_id = view
                            .outputs
                            .contains(ViewOutputs::OBJECT_ID)
                            .then(|| acquire(graph::SENSOR_OBJECT_ID_FORMAT));
                        let depth_attachment = Some(acquire(DEPTH_FORMAT));
                        self.compiled_views.push(CompiledView {
                            request: view,
                            external: None,
                            color,
                            depth,
                            object_id,
                            depth_attachment,
                        });
                    }
                }
            }
        }

        {
            profiling::scope!("upload instances and views");
            if !self.instances.is_empty() {
                self.queue.write_buffer(
                    &self.instance_buffer,
                    0,
                    bytemuck::cast_slice(&self.instances),
                );
            }
            for (index, view) in frame.views().iter().enumerate() {
                let view_proj = view
                    .camera
                    .view_projection(view.width as f32 / view.height as f32);
                let inv_view_proj = view_proj.inverse();
                let globals = Globals {
                    view: view.camera.view().to_cols_array_2d(),
                    view_projection: view_proj.to_cols_array_2d(),
                    inv_view_projection: inv_view_proj.to_cols_array_2d(),
                    camera_position: [
                        view.camera.eye.x,
                        view.camera.eye.y,
                        view.camera.eye.z,
                        1.0,
                    ],
                    light_direction: [0.55, 0.78, 0.30, 0.0],
                    light_color: [1.40, 1.35, 1.25, 1.0],
                    ambient_color: [0.42, 0.46, 0.52, 1.0],
                };
                self.queue.write_buffer(
                    &self.globals,
                    index as u64 * u64::from(self.globals_stride),
                    bytemuck::bytes_of(&globals),
                );
            }
        }

        profiling::scope!("encode frame graph");
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame graph encoder"),
            });
        let mut readbacks = Vec::new();
        for (view_index, compiled) in self.compiled_views.iter().copied().enumerate() {
            let dynamic_offset = view_index as u32 * self.globals_stride;
            match compiled.request.kind {
                ViewKind::Display => {
                    profiling::scope!("display pass");
                    let external = external_views[compiled.external.expect("display target")];
                    let color_attachments = [Some(color_attachment(
                        external.color,
                        wgpu::Color {
                            r: 0.015,
                            g: 0.025,
                            b: 0.045,
                            a: 1.0,
                        },
                    ))];
                    let mut pass = begin_pass(
                        &mut encoder,
                        "display pass",
                        &color_attachments,
                        external.depth,
                    );
                    encode_batches(
                        &mut pass,
                        &self.display_pipeline,
                        &self.globals_bind_group,
                        dynamic_offset,
                        &self.instance_buffer,
                        &self.meshes,
                        &self.batches,
                    );
                    pass.set_pipeline(&self.skybox_pipeline);
                    pass.set_bind_group(0, &self.globals_bind_group, &[dynamic_offset]);
                    pass.set_bind_group(1, &self.skybox_bind_group, &[]);
                    pass.draw(0..3, 0..1);
                }
                ViewKind::Sensor => {
                    profiling::scope!("sensor pass");
                    let color_attachments = [
                        compiled.color.map(|index| {
                            color_attachment(
                                self.texture_pool.view(index),
                                wgpu::Color::TRANSPARENT,
                            )
                        }),
                        compiled.depth.map(|index| {
                            color_attachment(
                                self.texture_pool.view(index),
                                wgpu::Color {
                                    r: compiled.request.camera.far as f64,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 0.0,
                                },
                            )
                        }),
                        compiled.object_id.map(|index| {
                            color_attachment(
                                self.texture_pool.view(index),
                                wgpu::Color::TRANSPARENT,
                            )
                        }),
                    ];
                    let depth_attachment = self
                        .texture_pool
                        .view(compiled.depth_attachment.expect("sensor depth attachment"));
                    let mut pass = begin_pass(
                        &mut encoder,
                        "sensor pass",
                        &color_attachments,
                        depth_attachment,
                    );
                    let pipeline = self.sensor_pipelines[compiled.request.outputs.bits() as usize]
                        .as_ref()
                        .expect("sensor views request at least one output");
                    encode_batches(
                        &mut pass,
                        pipeline,
                        &self.globals_bind_group,
                        dynamic_offset,
                        &self.instance_buffer,
                        &self.meshes,
                        &self.batches,
                    );
                    drop(pass);

                    {
                        profiling::scope!("schedule sensor readbacks");
                        for (output, texture_index) in [
                            (OutputKind::Color, compiled.color),
                            (OutputKind::Depth, compiled.depth),
                            (OutputKind::ObjectId, compiled.object_id),
                        ] {
                            let Some(texture_index) = texture_index else {
                                continue;
                            };
                            let handle = self
                                .readbacks
                                .schedule(
                                    &self.device,
                                    &mut encoder,
                                    compiled.request.key,
                                    output,
                                    self.texture_pool.texture(texture_index),
                                    compiled.request.width,
                                    compiled.request.height,
                                )
                                .ok_or(RendererError::ReadbackRingFull {
                                    view: compiled.request.key,
                                    output,
                                })?;
                            readbacks.push(handle);
                        }
                    }
                }
            }
        }
        let submission = self.queue.submit(Some(encoder.finish()));
        self.readbacks.begin_mapping(&readbacks);
        Ok(FrameSubmission {
            submission,
            readbacks,
        })
    }

    pub fn poll_readback(
        &mut self,
        handle: ReadbackHandle,
    ) -> Result<Option<ReadbackData>, RendererError> {
        self.device
            .poll(wgpu::PollType::Poll)
            .map_err(|_| RendererError::ReadbackFailed)?;
        self.readbacks
            .poll(handle)
            .map_err(|_| RendererError::InvalidReadbackHandle)
    }

    fn ensure_instance_capacity(&mut self, required: usize) {
        if required <= self.instance_capacity {
            return;
        }
        profiling::scope!("grow instance buffer");
        self.instance_capacity = required.next_power_of_two();
        self.instance_buffer = create_instance_buffer(&self.device, self.instance_capacity);
    }

    fn ensure_globals_capacity(&mut self, required: usize) {
        if required <= self.globals_capacity {
            return;
        }
        profiling::scope!("grow view uniforms");
        self.globals_capacity = required.next_power_of_two();
        self.globals =
            create_globals_buffer(&self.device, self.globals_capacity, self.globals_stride);
        self.globals_bind_group =
            create_globals_bind_group(&self.device, &self.globals_layout, &self.globals);
    }
}

pub struct DepthTarget {
    view: wgpu::TextureView,
}

impl DepthTarget {
    pub fn new(renderer: &Renderer, width: u32, height: u32) -> Self {
        let texture = create_target(
            renderer.device(),
            width,
            height,
            DEPTH_FORMAT,
            "scene depth",
        );
        Self {
            view: texture.create_view(&Default::default()),
        }
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }
}

pub struct OffscreenTarget {
    width: u32,
    height: u32,
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth: DepthTarget,
}

impl OffscreenTarget {
    pub fn new(renderer: &Renderer, width: u32, height: u32) -> Self {
        assert!(
            width > 0 && height > 0,
            "render dimensions must be non-zero"
        );
        let color = create_target(
            renderer.device(),
            width,
            height,
            renderer.color_format(),
            "scene color",
        );
        let color_view = color.create_view(&Default::default());
        let depth = DepthTarget::new(renderer, width, height);
        Self {
            width,
            height,
            color,
            color_view,
            depth,
        }
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn render_target(&self) -> RenderTarget<'_> {
        RenderTarget {
            color: &self.color_view,
            depth: self.depth.view(),
            width: self.width,
            height: self.height,
        }
    }

    pub async fn read_rgba(&self, renderer: &Renderer) -> Result<Vec<u8>, RendererError> {
        profiling::function_scope!();
        let unpadded_bytes_per_row = self.width * 4;
        let padded_bytes_per_row =
            unpadded_bytes_per_row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let output = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene readback"),
            size: u64::from(padded_bytes_per_row) * u64::from(self.height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = renderer
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scene readback encoder"),
            });
        encoder.copy_texture_to_buffer(
            self.color.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &output,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        renderer.queue.submit(Some(encoder.finish()));

        let slice = output.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        renderer
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|_| RendererError::ReadbackFailed)?;
        receiver
            .recv()
            .map_err(|_| RendererError::ReadbackFailed)?
            .map_err(|_| RendererError::ReadbackFailed)?;
        let mapped = slice
            .get_mapped_range()
            .map_err(|_| RendererError::ReadbackFailed)?;
        let mut pixels = Vec::with_capacity((unpadded_bytes_per_row * self.height) as usize);
        for row in mapped.chunks_exact(padded_bytes_per_row as usize) {
            pixels.extend_from_slice(&row[..unpadded_bytes_per_row as usize]);
        }
        drop(mapped);
        output.unmap();
        Ok(pixels)
    }
}

fn create_target(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    label: &str,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | if format != DEPTH_FORMAT {
                wgpu::TextureUsages::COPY_SRC
            } else {
                wgpu::TextureUsages::empty()
            },
        view_formats: &[],
    })
}

fn create_globals_buffer(device: &wgpu::Device, capacity: usize, stride: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("view globals"),
        size: capacity as u64 * u64::from(stride),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_globals_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scene globals bind group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: 0,
                size: wgpu::BufferSize::new(size_of::<Globals>() as u64),
            }),
        }],
    })
}

fn color_target(format: wgpu::TextureFormat) -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    }
}

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    label: &str,
    fragment_entry: &str,
    targets: &[Option<wgpu::ColorTargetState>],
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[Some(vertex_layout()), Some(instance_layout())],
        },
        primitive: wgpu::PrimitiveState {
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fragment_entry),
            compilation_options: Default::default(),
            targets,
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn color_attachment(
    view: &wgpu::TextureView,
    clear: wgpu::Color,
) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(clear),
            store: wgpu::StoreOp::Store,
        },
    }
}

fn begin_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    label: &'a str,
    color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>],
    depth: &'a wgpu::TextureView,
) -> wgpu::RenderPass<'a> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments,
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Discard,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

fn encode_batches<'a>(
    pass: &mut wgpu::RenderPass<'a>,
    pipeline: &'a wgpu::RenderPipeline,
    globals: &'a wgpu::BindGroup,
    globals_offset: u32,
    instances: &'a wgpu::Buffer,
    meshes: &'a ResourceRegistry<Mesh>,
    batches: &'a [DrawBatch],
) {
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, globals, &[globals_offset]);
    pass.set_vertex_buffer(1, instances.slice(..));
    for batch in batches {
        let mesh = meshes
            .get(batch.mesh)
            .expect("mesh handles were validated before encoding");
        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(
            0..mesh.index_count,
            0,
            batch.first_instance..batch.first_instance + batch.instance_count,
        );
    }
}

fn create_instance_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("primitive instances"),
        size: (capacity * size_of::<Instance>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4, 7 => Uint32];

fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: size_of::<MeshVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &VERTEX_ATTRIBUTES,
    }
}

fn instance_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: size_of::<Instance>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &INSTANCE_ATTRIBUTES,
    }
}
fn create_skybox_texture_and_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    width: u32,
    height: u32,
    rgba_data: &[u8],
) -> (wgpu::Texture, wgpu::BindGroup) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("skybox texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba_data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("skybox sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("skybox bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    (texture, bind_group)
}

fn generate_default_skybox_rgba(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        let v = y as f32 / (height - 1) as f32;
        let (r, g, b) = if v < 0.5 {
            let t = v / 0.5;
            (
                0.15 + (0.80 - 0.15) * t.powf(0.6),
                0.40 + (0.85 - 0.40) * t.powf(0.6),
                0.85 + (0.95 - 0.85) * t.powf(0.6),
            )
        } else {
            let t = (v - 0.5) / 0.5;
            (
                0.80 + (0.08 - 0.80) * t.powf(0.4),
                0.85 + (0.10 - 0.85) * t.powf(0.4),
                0.95 + (0.14 - 0.95) * t.powf(0.4),
            )
        };
        for _x in 0..width {
            pixels.push((r.clamp(0.0, 1.0) * 255.0) as u8);
            pixels.push((g.clamp(0.0, 1.0) * 255.0) as u8);
            pixels.push((b.clamp(0.0, 1.0) * 255.0) as u8);
            pixels.push(255);
        }
    }
    pixels
}
fn dynamic_image_to_srgb_rgba8(img: &image::DynamicImage) -> image::RgbaImage {
    let aces_tonemap = |x: f32| -> f32 {
        let a = 2.51;
        let b = 0.03;
        let c = 2.43;
        let d = 0.59;
        let e = 0.14;
        ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
    };

    let to_srgb = |c: f32| -> u8 {
        let s = if c <= 0.0031308 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        (s.clamp(0.0, 1.0) * 255.0).round() as u8
    };

    match img {
        image::DynamicImage::ImageRgb32F(f_img) => {
            let (width, height) = (f_img.width(), f_img.height());
            let mut out = image::RgbaImage::new(width, height);
            for (x, y, pixel) in f_img.enumerate_pixels() {
                let r = aces_tonemap(pixel[0].max(0.0));
                let g = aces_tonemap(pixel[1].max(0.0));
                let b = aces_tonemap(pixel[2].max(0.0));
                out.put_pixel(
                    x,
                    y,
                    image::Rgba([to_srgb(r), to_srgb(g), to_srgb(b), 255]),
                );
            }
            out
        }
        image::DynamicImage::ImageRgba32F(f_img) => {
            let (width, height) = (f_img.width(), f_img.height());
            let mut out = image::RgbaImage::new(width, height);
            for (x, y, pixel) in f_img.enumerate_pixels() {
                let r = aces_tonemap(pixel[0].max(0.0));
                let g = aces_tonemap(pixel[1].max(0.0));
                let b = aces_tonemap(pixel[2].max(0.0));
                let a = (pixel[3].clamp(0.0, 1.0) * 255.0).round() as u8;
                out.put_pixel(
                    x,
                    y,
                    image::Rgba([to_srgb(r), to_srgb(g), to_srgb(b), a]),
                );
            }
            out
        }
        other => other.to_rgba8(),
    }
}
