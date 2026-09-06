pub struct DepthRange {
    buffer: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    bindings: Option<wgpu::BindGroup>,
}

impl DepthRange {
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visible depth range"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("visible depth reduction"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(8),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("visible depth reduction"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("visible depth reduction"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../depth_range.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("visible depth reduction"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("reduce"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            buffer,
            pipeline,
            layout,
            bindings: None,
        }
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn bind(
        &mut self,
        device: &wgpu::Device,
        depth: &wgpu::TextureView,
        ids: &wgpu::TextureView,
    ) {
        self.bindings = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visible depth inputs"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(ids),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buffer.as_entire_binding(),
                },
            ],
        }));
    }

    pub fn encode(&self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, size: [u32; 2]) {
        queue.write_buffer(
            &self.buffer,
            0,
            bytemuck::cast_slice(&[f32::INFINITY.to_bits(), 0u32]),
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("visible depth reduction"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(
            0,
            self.bindings.as_ref().expect("depth inputs initialized"),
            &[],
        );
        pass.dispatch_workgroups(size[0].div_ceil(16), size[1].div_ceil(16), 1);
    }
}
