use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub vertical_fov_radians: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    fn view_projection(self, aspect: f32) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye, self.target, self.up);
        let projection =
            Mat4::perspective_rh(self.vertical_fov_radians, aspect, self.near, self.far);
        projection * view
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Cube {
    pub transform: Mat4,
    pub color: [f32; 4],
}

#[derive(Debug)]
pub enum RendererError {
    NoAdapter(wgpu::RequestAdapterError),
    RequestDevice(wgpu::RequestDeviceError),
    TooManyInstances(usize),
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
            Self::ReadbackFailed => write!(f, "GPU readback failed"),
        }
    }
}

impl std::error::Error for RendererError {}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

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

pub struct Renderer {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    color_format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    globals: wgpu::Buffer,
    globals_bind_group: wgpu::BindGroup,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    instance_capacity: usize,
    instances: Vec<Instance>,
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
            label: Some("cube pipeline layout"),
            bind_group_layouts: &[Some(&globals_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("instanced cube pipeline"),
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
        let (vertices_data, indices_data) = cube_geometry();
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cube vertices"),
            contents: bytemuck::cast_slice(&vertices_data),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cube indices"),
            contents: bytemuck::cast_slice(&indices_data),
            usage: wgpu::BufferUsages::INDEX,
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
            vertices,
            indices,
            instance_buffer,
            instance_capacity,
            instances: Vec::with_capacity(instance_capacity),
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

    pub fn render(
        &mut self,
        target: RenderTarget<'_>,
        camera: Camera,
        cubes: &[Cube],
    ) -> Result<wgpu::SubmissionIndex, RendererError> {
        assert!(
            target.width > 0 && target.height > 0,
            "render dimensions must be non-zero"
        );
        let instance_count =
            u32::try_from(cubes.len()).map_err(|_| RendererError::TooManyInstances(cubes.len()))?;
        self.ensure_instance_capacity(cubes.len());
        self.instances.clear();
        self.instances.extend(cubes.iter().map(|cube| Instance {
            model: cube.transform.to_cols_array_2d(),
            color: cube.color,
        }));
        if !self.instances.is_empty() {
            self.queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&self.instances),
            );
        }
        let globals = Globals {
            view_projection: camera
                .view_projection(target.width as f32 / target.height as f32)
                .to_cols_array_2d(),
            light_direction: [0.35, 0.8, 0.45, 0.0],
        };
        self.queue
            .write_buffer(&self.globals, 0, bytemuck::bytes_of(&globals));

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
            pass.set_vertex_buffer(0, self.vertices.slice(..));
            pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..36, 0, 0..instance_count);
        }
        Ok(self.queue.submit(Some(encoder.finish())))
    }

    fn ensure_instance_capacity(&mut self, required: usize) {
        if required <= self.instance_capacity {
            return;
        }
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
        label: Some("cube instances"),
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
        array_stride: size_of::<Vertex>() as u64,
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

fn cube_geometry() -> ([Vertex; 24], [u16; 36]) {
    let mut vertices = [Vertex {
        position: [0.0; 3],
        normal: [0.0; 3],
    }; 24];
    let faces = [
        (
            [1.0, 0.0, 0.0],
            [
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, -0.5],
            ],
        ),
        (
            [-1.0, 0.0, 0.0],
            [
                [-0.5, -0.5, 0.5],
                [-0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, 0.5, 0.5],
            ],
        ),
        (
            [0.0, 1.0, 0.0],
            [
                [-0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ],
        ),
        (
            [0.0, -1.0, 0.0],
            [
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, -0.5, -0.5],
                [-0.5, -0.5, -0.5],
            ],
        ),
        (
            [0.0, 0.0, 1.0],
            [
                [0.5, -0.5, 0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ],
        ),
        (
            [0.0, 0.0, -1.0],
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
            ],
        ),
    ];
    for (face_index, (normal, positions)) in faces.into_iter().enumerate() {
        for (corner, position) in positions.into_iter().enumerate() {
            vertices[face_index * 4 + corner] = Vertex { position, normal };
        }
    }
    let mut indices = [0_u16; 36];
    for face in 0..6_u16 {
        let base = face * 4;
        indices[(face as usize) * 6..(face as usize + 1) * 6].copy_from_slice(&[
            base,
            base + 2,
            base + 1,
            base,
            base + 3,
            base + 2,
        ]);
    }
    (vertices, indices)
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::cube_geometry;

    #[test]
    fn cube_triangles_wind_counter_clockwise_from_outside() {
        let (vertices, indices) = cube_geometry();
        for triangle in indices.chunks_exact(3) {
            let a = Vec3::from_array(vertices[triangle[0] as usize].position);
            let b = Vec3::from_array(vertices[triangle[1] as usize].position);
            let c = Vec3::from_array(vertices[triangle[2] as usize].position);
            let expected_normal =
                Vec3::from_array(vertices[triangle[0] as usize].normal);
            let geometric_normal = (b - a).cross(c - a);

            assert!(
                geometric_normal.dot(expected_normal) > 0.0,
                "triangle {triangle:?} faces into the cube"
            );
        }
    }
}
