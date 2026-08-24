use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

mod camera;
mod frame;
mod primitive;
mod resource;

pub use camera::Camera;
pub use frame::Frame;
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
        }
    }
}

impl std::error::Error for RendererError {}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Instance {
    model: [[f32; 4]; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    view_projection: [[f32; 4]; 4],
    light_direction: [f32; 4],
}

#[derive(Clone, Copy)]
struct DrawBatch {
    mesh: MeshHandle,
    first_instance: u32,
    instance_count: u32,
}

pub struct Renderer {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    color_format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    globals: wgpu::Buffer,
    globals_bind_group: wgpu::BindGroup,
    meshes: ResourceRegistry<Mesh>,
    instance_buffer: wgpu::Buffer,
    instance_capacity: usize,
    instances: Vec<Instance>,
    batches: Vec<DrawBatch>,
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
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("sim-graphics device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(RendererError::RequestDevice)?;

        let globals = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scene globals"),
            contents: bytemuck::bytes_of(&Globals::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let globals_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene globals layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene globals bind group"),
            layout: &globals_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals.as_entire_binding(),
            }],
        });
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("primitive pipeline layout"),
            bind_group_layouts: &[Some(&globals_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("instanced primitive pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
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
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
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
            pipeline,
            globals,
            globals_bind_group,
            meshes: ResourceRegistry::new(),
            instance_buffer,
            instance_capacity,
            instances: Vec::with_capacity(instance_capacity),
            batches: Vec::new(),
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

    pub fn remove_mesh(&mut self, handle: MeshHandle) -> bool {
        self.meshes.remove(handle).is_some()
    }

    pub fn render(
        &mut self,
        target: RenderTarget<'_>,
        frame: &Frame,
    ) -> Result<wgpu::SubmissionIndex, RendererError> {
        profiling::function_scope!();
        assert!(
            target.width > 0 && target.height > 0,
            "render dimensions must be non-zero"
        );
        {
            profiling::scope!("prepare frame");
            let primitive_count = frame.primitives().len();
            u32::try_from(primitive_count)
                .map_err(|_| RendererError::TooManyInstances(primitive_count))?;
            self.ensure_instance_capacity(primitive_count);
            self.instances.clear();
            self.batches.clear();

            for primitive in frame.primitives() {
                if self.meshes.get(primitive.mesh).is_none() {
                    return Err(RendererError::InvalidMeshHandle(primitive.mesh));
                }
                self.instances.push(Instance {
                    model: primitive.transform.to_cols_array_2d(),
                    color: primitive.color,
                });
                match self.batches.last_mut() {
                    Some(batch) if batch.mesh == primitive.mesh => {
                        batch.instance_count += 1;
                    }
                    _ => self.batches.push(DrawBatch {
                        mesh: primitive.mesh,
                        first_instance: (self.instances.len() - 1) as u32,
                        instance_count: 1,
                    }),
                }
            }
        }
        {
            profiling::scope!("upload frame");
            if !self.instances.is_empty() {
                self.queue.write_buffer(
                    &self.instance_buffer,
                    0,
                    bytemuck::cast_slice(&self.instances),
                );
            }
            let globals = Globals {
                view_projection: frame
                    .camera()
                    .view_projection(target.width as f32 / target.height as f32)
                    .to_cols_array_2d(),
                light_direction: [0.35, 0.8, 0.45, 0.0],
            };
            self.queue
                .write_buffer(&self.globals, 0, bytemuck::bytes_of(&globals));
        }

        profiling::scope!("encode and submit");
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scene encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target.color,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.015,
                            g: 0.025,
                            b: 0.045,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: target.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.globals_bind_group, &[]);
            pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            for batch in &self.batches {
                let mesh = self
                    .meshes
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
        Ok(self.queue.submit(Some(encoder.finish())))
    }

    fn ensure_instance_capacity(&mut self, required: usize) {
        if required <= self.instance_capacity {
            return;
        }
        profiling::scope!("grow instance buffer");
        self.instance_capacity = required.next_power_of_two();
        self.instance_buffer = create_instance_buffer(&self.device, self.instance_capacity);
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
const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4];

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
