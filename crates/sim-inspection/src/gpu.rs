use super::*;
use bytemuck::{Pod, Zeroable};
use sim_graphics::OutputKind;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Quad {
    rect: [f32; 4],
    color: [f32; 4],
    params: [f32; 4],
}

/// GPU-resident interactive presentation. Only `capture` maps sensor data to the CPU.
pub struct GpuInspector {
    inspector: Inspector,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    globals: wgpu::Buffer,
    atlas: wgpu::Texture,
    observer: wgpu::Texture,
    sensor: Option<[wgpu::Texture; 3]>,
    bindings: Option<wgpu::BindGroup>,
    dimensions: [u32; 2],
    vertices: wgpu::Buffer,
    capacity: usize,
    quads: Vec<Quad>,
}

fn texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("inspection persistent image"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}
fn vertex_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("inspection grow-only quads"),
        size: (capacity * size_of::<Quad>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl GpuInspector {
    pub async fn new(
        instance: &wgpu::Instance,
        surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Self> {
        let mut renderer = Renderer::new(instance, surface).await?;
        let cube = renderer.register_mesh(MeshData::cube())?;
        let device = renderer.device();
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("inspection globals"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        for binding in 1..=5 {
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
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("inspection images"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("inspection compositor"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("inspection compositor"),
            source: wgpu::ShaderSource::Wgsl(include_str!("inspector.wgsl").into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("inspection compositor"), layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader, entry_point: Some("vs_main"), compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<Quad>() as u64, step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x4],
                })],
            },
            primitive: Default::default(), depth_stencil: None, multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader, entry_point: Some("fs_main"), compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: renderer.color_format(), blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }), multiview_mask: None, cache: None,
        });
        let atlas = texture(device, 128, 64, wgpu::TextureFormat::R8Unorm);
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
        renderer.queue().write_texture(
            atlas.as_image_copy(),
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(128),
                rows_per_image: Some(64),
            },
            atlas.size(),
        );
        let observer = texture(device, 640, 480, wgpu::TextureFormat::Rgba8Unorm);
        let capacity = 2048;
        let vertices = vertex_buffer(device, capacity);
        Ok(Self {
            inspector: Inspector {
                renderer,
                cube,
                frame: Frame::with_capacity(64, 1),
                healthy: true,
            },
            pipeline,
            layout,
            globals,
            atlas,
            observer,
            sensor: None,
            bindings: None,
            dimensions: [0, 0],
            vertices,
            capacity,
            quads: Vec::with_capacity(capacity),
        })
    }
    pub fn format(&self) -> wgpu::TextureFormat {
        self.inspector.renderer.color_format()
    }
    pub fn device(&self) -> &wgpu::Device {
        self.inspector.renderer.device()
    }
    pub fn queue(&self) -> &wgpu::Queue {
        self.inspector.renderer.queue()
    }
    pub fn capture(&mut self, scene: &SceneConfig) -> Result<Capture> {
        self.inspector.capture(scene)
    }

    fn resize_sensor(&mut self, width: u32, height: u32) {
        if self.dimensions == [width, height] {
            return;
        }
        let sensor = [
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::R32Float,
            wgpu::TextureFormat::R32Uint,
        ]
        .map(|format| texture(self.device(), width, height, format));
        let views = [
            &sensor[0],
            &sensor[1],
            &sensor[2],
            &self.observer,
            &self.atlas,
        ]
        .map(|texture| texture.create_view(&Default::default()));
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: self.globals.as_entire_binding(),
        }];
        for (index, view) in views.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: index as u32 + 1,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
        self.bindings = Some(self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("inspection images"),
            layout: &self.layout,
            entries: &entries,
        }));
        self.sensor = Some(sensor);
        self.dimensions = [width, height];
    }

    pub fn render(
        &mut self,
        scene: &SceneConfig,
        target: &wgpu::TextureView,
        width: u32,
        height: u32,
        status: &[String],
    ) -> Result<()> {
        scene.validate()?;
        ensure!(
            width > 0 && height > 0,
            "inspection target must be nonempty"
        );
        self.resize_sensor(scene.sensor.width, scene.sensor.height);
        self.inspector.scene(scene);
        let s = &scene.sensor;
        self.inspector.frame.add_view(RenderView {
            key: ViewKey(1),
            kind: ViewKind::Sensor,
            camera: sensor_camera(s),
            width: s.width,
            height: s.height,
            outputs: ViewOutputs::COLOR | ViewOutputs::DEPTH | ViewOutputs::OBJECT_ID,
        });
        self.inspector
            .renderer
            .execute_gpu(&self.inspector.frame, &[])?;
        // Submit these copies before the observer graph can reuse any transient textures.
        let mut copies = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("preserve synchronized sensor outputs"),
            });
        for (kind, dest) in [OutputKind::Color, OutputKind::Depth, OutputKind::ObjectId]
            .into_iter()
            .zip(self.sensor.as_ref().unwrap())
        {
            let source = self
                .inspector
                .renderer
                .output_texture(ViewKey(1), kind)
                .context("missing sensor graph output")?;
            copies.copy_texture_to_texture(
                source.as_image_copy(),
                dest.as_image_copy(),
                dest.size(),
            );
        }
        self.queue().submit(Some(copies.finish()));
        let observer_camera = self.inspector.observer_scene(scene);
        self.inspector.frame.add_view(RenderView {
            key: ViewKey(2),
            kind: ViewKind::Sensor,
            camera: observer_camera,
            width: 640,
            height: 480,
            outputs: ViewOutputs::COLOR,
        });
        self.inspector
            .renderer
            .execute_gpu(&self.inspector.frame, &[])?;
        self.compose(scene, observer_camera, width, height, status);
        if self.quads.len() > self.capacity {
            self.capacity = self.quads.len().next_power_of_two();
            self.vertices = vertex_buffer(self.device(), self.capacity);
        }
        self.queue()
            .write_buffer(&self.vertices, 0, bytemuck::cast_slice(&self.quads));
        self.queue().write_buffer(
            &self.globals,
            0,
            bytemuck::cast_slice(&[
                width as f32,
                height as f32,
                s.near,
                s.far,
                if self.format().is_srgb() { 1.0 } else { 0.0 },
                0.0,
                0.0,
                0.0,
            ]),
        );
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("inspection presentation"),
            });
        let source = self
            .inspector
            .renderer
            .output_texture(ViewKey(2), OutputKind::Color)
            .context("missing observer graph output")?;
        encoder.copy_texture_to_texture(
            source.as_image_copy(),
            self.observer.as_image_copy(),
            self.observer.size(),
        );
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("inspection four panels and annotations"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
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
            pass.set_bind_group(0, self.bindings.as_ref().unwrap(), &[]);
            pass.set_vertex_buffer(0, self.vertices.slice(..));
            pass.draw(0..6, 0..self.quads.len() as u32);
        }
        self.queue().submit(Some(encoder.finish()));
        Ok(())
    }

    fn quad(&mut self, rect: [f32; 4], color: [u8; 4], mode: f32, glyph: f32, angle: f32) {
        self.quads.push(Quad {
            rect,
            color: color.map(|v| v as f32 / 255.0),
            params: [mode, glyph, angle, 0.0],
        });
    }
    fn text(&mut self, x: f32, y: f32, value: &str, color: [u8; 4]) {
        for (i, ch) in value.chars().enumerate() {
            if x + (i + 1) as f32 * 16.0 > 1280.0 {
                break;
            }
            self.quad(
                [x + i as f32 * 16.0, y, 16.0, 16.0],
                color,
                4.0,
                (if ch.is_ascii() { ch as u32 } else { 63 }) as f32,
                0.0,
            );
        }
    }
    fn callout(
        &mut self,
        camera: Camera,
        point: Vec3,
        label: &str,
        color: [u8; 4],
        anchor: [f32; 2],
    ) {
        let clip = camera.view_projection(640.0 / 480.0) * point.extend(1.0);
        if clip.w <= 0.0 {
            return;
        }
        let ndc = clip.truncate() / clip.w;
        if ndc.x.abs() > 1.0 || ndc.y.abs() > 1.0 || !(0.0..=1.0).contains(&ndc.z) {
            return;
        }
        let end = glam::Vec2::new(640.0 + (ndc.x + 1.0) * 320.0, 544.0 + (1.0 - ndc.y) * 240.0);
        let start = glam::Vec2::new(
            (anchor[0] + label.chars().count() as f32 * 16.0).min(1279.0),
            anchor[1] + 6.0,
        );
        let delta = end - start;
        self.quad(
            [start.x, start.y, delta.length(), 1.0],
            color,
            5.0,
            0.0,
            delta.y.atan2(delta.x),
        );
        self.text(anchor[0], anchor[1], label, color);
    }
    fn compose(
        &mut self,
        scene: &SceneConfig,
        observer: Camera,
        width: u32,
        height: u32,
        status: &[String],
    ) {
        self.quads.clear();
        let status_rows: usize = status
            .iter()
            .map(|line| line.chars().count().max(1).div_ceil(79))
            .sum();
        let canvas_height = 1160.0 + (scene.objects.len() + status_rows) as f32 * 20.0;
        self.quad(
            [0.0, 0.0, 1280.0, canvas_height],
            [16, 19, 25, 255],
            5.0,
            0.0,
            0.0,
        );
        let s = &scene.sensor;
        for (index, title) in [
            "RGB | sRGB preview (linear truth)",
            "DEPTH | optical Z: near -> far",
            "INSTANCE IDs | uint32 truth",
            "OVERVIEW | frustum + world axes",
        ]
        .iter()
        .enumerate()
        {
            let x = (index % 2) as f32 * 640.0;
            let y = (index / 2) as f32 * 512.0;
            self.text(x + 8.0, y + 10.0, title, [235, 240, 250, 255]);
            let (pw, ph) = if index == 3 {
                (640.0, 480.0)
            } else {
                (s.width as f32, s.height as f32)
            };
            let scale = (640.0 / pw).min(480.0 / ph);
            self.quad(
                [
                    x + (640.0 - pw * scale) * 0.5,
                    y + 32.0 + (480.0 - ph * scale) * 0.5,
                    pw * scale,
                    ph * scale,
                ],
                [255; 4],
                index as f32,
                0.0,
                0.0,
            );
        }
        for (index, object) in scene.objects.iter().enumerate() {
            self.callout(
                observer,
                object_center(object),
                &format!("{} {}", object.id, object.name),
                id_color(object.id),
                [648.0, 544.0 + (16.0 + index as f32 * 18.0).min(420.0)],
            );
        }
        self.callout(
            observer,
            s.eye.into(),
            "SENSOR",
            [255, 220, 70, 255],
            [648.0, 996.0],
        );
        for (index, (point, label, color)) in [
            (Vec3::X * 2.0, "+X", [255, 80, 80, 255]),
            (Vec3::Y * 2.0, "+Y", [80, 255, 80, 255]),
            (Vec3::Z * 2.0, "+Z", [90, 130, 255, 255]),
        ]
        .into_iter()
        .enumerate()
        {
            self.callout(
                observer,
                point,
                label,
                color,
                [1210.0, 560.0 + index as f32 * 20.0],
            );
        }
        let k = s.intrinsics();
        let labels = [
            format!(
                "{}x{} | fx=fy {:.2} | cx={:.2} cy={:.2} | FOVy {:.2}deg",
                s.width,
                s.height,
                k[0][0],
                k[0][2],
                k[1][2],
                s.vertical_fov_radians.to_degrees()
            ),
            format!(
                "Near {:.3}m Far {:.3}m | RH world: Y up (metres)",
                s.near, s.far
            ),
            "Optical: X right / Y down / Z forward | background ID=0 depth=far".into(),
            "RGB truth: linear UNORM; preview: sRGB. Frustum: true near/far.".into(),
            "Gizmos are observer-only. Palette colors are not instance-ID data.".into(),
        ];
        for (i, label) in labels.iter().enumerate() {
            self.text(8.0, 1032.0 + i as f32 * 20.0, label, [210, 215, 225, 255]);
        }
        for (i, object) in scene.objects.iter().enumerate() {
            self.text(
                8.0,
                1136.0 + i as f32 * 20.0,
                &format!("ID {}: {}", object.id, object.name),
                id_color(object.id),
            );
        }
        let mut row = scene.objects.len();
        for line in status {
            for (i, ch) in line.chars().enumerate() {
                let x = 8.0 + (i % 79) as f32 * 16.0;
                let y = 1136.0 + (row + i / 79) as f32 * 20.0;
                self.quad(
                    [x, y, 16.0, 16.0],
                    [235, 240, 250, 255],
                    4.0,
                    (if ch.is_ascii() { ch as u32 } else { 63 }) as f32,
                    0.0,
                );
            }
            row += line.chars().count().max(1).div_ceil(79);
        }
        let scale = (width as f32 / 1280.0).min(height as f32 / canvas_height);
        let offset = [
            (width as f32 - 1280.0 * scale) * 0.5,
            (height as f32 - canvas_height * scale) * 0.5,
        ];
        for quad in &mut self.quads {
            quad.rect[0] = quad.rect[0] * scale + offset[0];
            quad.rect[1] = quad.rect[1] * scale + offset[1];
            quad.rect[2] *= scale;
            quad.rect[3] *= scale;
        }
    }
}
