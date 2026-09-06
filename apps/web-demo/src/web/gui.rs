use bytemuck::{Pod, Zeroable};
use font8x8::UnicodeFonts;

use super::controls::Button;

const MAX_BUTTONS: usize = 9;
const MAX_LABEL_CHARS: usize = 32;
const QUAD_CAPACITY: usize = MAX_BUTTONS * (5 + MAX_LABEL_CHARS);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Quad {
    rect: [f32; 4],
    color: [f32; 4],
    glyph: f32,
}

pub struct Gui {
    pipeline: wgpu::RenderPipeline,
    bindings: wgpu::BindGroup,
    globals: wgpu::Buffer,
    instances: wgpu::Buffer,
    quads: Vec<Quad>,
    surface_srgb: bool,
}

impl Gui {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("canvas GUI globals"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("canvas GUI persistent ASCII atlas"),
            size: wgpu::Extent3d {
                width: 128,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let mut pixels = [0u8; 128 * 64];
        for ch in 0u8..128 {
            if let Some(glyph) = font8x8::BASIC_FONTS.get(ch as char) {
                for (y, row) in glyph.iter().enumerate() {
                    for x in 0..8 {
                        pixels[((ch as usize / 16) * 8 + y) * 128 + (ch as usize % 16) * 8 + x] =
                            if row & (1 << x) != 0 { 255 } else { 0 };
                    }
                }
            }
        }
        queue.write_texture(
            atlas.as_image_copy(),
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(128),
                rows_per_image: Some(64),
            },
            atlas.size(),
        );
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("canvas GUI bindings"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("canvas GUI persistent bindings"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &atlas.create_view(&Default::default()),
                    ),
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("canvas GUI pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("canvas GUI shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../gui.wgsl").into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("canvas GUI alpha overlay"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<Quad>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32],
                })],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("canvas GUI fixed-capacity quad instances"),
            size: (QUAD_CAPACITY * size_of::<Quad>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            bindings,
            globals,
            instances,
            quads: Vec::with_capacity(QUAD_CAPACITY),
            surface_srgb: format.is_srgb(),
        }
    }

    pub fn render(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        width: u32,
        height: u32,
        pixel_ratio: f32,
        buttons: &[Button],
    ) {
        if width == 0 || height == 0 {
            return;
        }
        self.quads.clear();
        for button in buttons.iter().take(MAX_BUTTONS) {
            let [x, y, w, h] = button.rect;
            if w <= 0.0 || h <= 0.0 {
                continue;
            }
            let border = if button.pressed {
                [1.0, 0.85, 0.55, 1.0]
            } else if button.active {
                [1.0, 0.65, 0.28, 1.0]
            } else {
                [0.48, 0.53, 0.61, 0.85]
            };
            let fill = if button.pressed {
                [0.40, 0.27, 0.14, 0.98]
            } else if button.active {
                [0.25, 0.17, 0.10, 0.96]
            } else {
                [0.055, 0.07, 0.10, 0.92]
            };
            let inset = (if button.pressed { 2.0 } else { 1.0 } * pixel_ratio)
                .min(w * 0.5)
                .min(h * 0.5);
            self.quads.push(Quad {
                rect: [x + inset, y + inset, w - 2.0 * inset, h - 2.0 * inset],
                color: fill,
                glyph: -1.0,
            });
            for rect in [
                [x, y, w, inset],
                [x, y + h - inset, w, inset],
                [x, y + inset, inset, h - 2.0 * inset],
                [x + w - inset, y + inset, inset, h - 2.0 * inset],
            ] {
                self.quads.push(Quad {
                    rect,
                    color: border,
                    glyph: -1.0,
                });
            }
            let advance = 8.0 * pixel_ratio;
            let count = button
                .label
                .chars()
                .count()
                .min(MAX_LABEL_CHARS)
                .min((w / advance).floor() as usize);
            let text_x = x + (w - count as f32 * advance) * 0.5;
            let glyph_height = (16.0 * pixel_ratio).min(h);
            let text_y = y + (h - glyph_height) * 0.5;
            for (index, ch) in button.label.chars().take(count).enumerate() {
                if ch == ' ' {
                    continue;
                }
                self.quads.push(Quad {
                    rect: [
                        text_x + index as f32 * advance,
                        text_y,
                        advance,
                        glyph_height,
                    ],
                    color: [1.0, 1.0, 1.0, 1.0],
                    glyph: if ch.is_ascii() {
                        ch as u32 as f32
                    } else {
                        63.0
                    },
                });
            }
        }
        if self.quads.is_empty() {
            return;
        }
        queue.write_buffer(
            &self.globals,
            0,
            bytemuck::cast_slice(&[
                width as f32,
                height as f32,
                u32::from(self.surface_srgb) as f32,
                0.0,
            ]),
        );
        queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(&self.quads));
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("canvas GUI presentation-only overlay"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bindings, &[]);
        pass.set_vertex_buffer(0, self.instances.slice(..));
        pass.draw(0..6, 0..self.quads.len() as u32);
    }
}
