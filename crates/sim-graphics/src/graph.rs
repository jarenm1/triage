use std::{collections::HashMap, sync::{Arc, atomic::{AtomicU8, Ordering}}};

use crate::{ViewId, ViewKey};

pub const SENSOR_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
pub const SENSOR_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
pub const SENSOR_OBJECT_ID_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OutputKind {
    Color,
    Depth,
    ObjectId,
}

#[derive(Clone, Copy, Debug)]
pub struct ExternalView<'a> {
    pub view: ViewId,
    pub color: &'a wgpu::TextureView,
    pub depth: &'a wgpu::TextureView,
}

#[derive(Debug)]
pub enum ReadbackData {
    Color(Vec<u8>),
    Depth(Vec<f32>),
    ObjectIds(Vec<u32>),
}

#[derive(Clone, Copy, Debug)]
pub struct ReadbackHandle {
    stream: StreamKey,
    slot: u8,
    generation: u32,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
}

impl ReadbackHandle {
    pub fn view(self) -> ViewKey {
        self.stream.view
    }

    pub fn output(self) -> OutputKind {
        self.stream.output
    }
}

#[derive(Debug)]
pub struct FrameSubmission {
    pub submission: wgpu::SubmissionIndex,
    pub readbacks: Vec<ReadbackHandle>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct StreamKey {
    view: ViewKey,
    output: OutputKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TextureKey {
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
    pub usage: wgpu::TextureUsages,
}

pub(crate) struct TexturePool {
    textures: Vec<PooledTexture>,
}

struct PooledTexture {
    key: TextureKey,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    in_use: bool,
}

impl TexturePool {
    pub fn new() -> Self {
        Self { textures: Vec::new() }
    }

    pub fn begin_frame(&mut self) {
        for texture in &mut self.textures {
            texture.in_use = false;
        }
    }

    pub fn acquire(&mut self, device: &wgpu::Device, key: TextureKey) -> usize {
        if let Some((index, texture)) = self
            .textures
            .iter_mut()
            .enumerate()
            .find(|(_, texture)| !texture.in_use && texture.key == key)
        {
            texture.in_use = true;
            return index;
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frame graph transient texture"),
            size: wgpu::Extent3d {
                width: key.width,
                height: key.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: key.format,
            usage: key.usage,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        self.textures.push(PooledTexture {
            key,
            texture,
            view,
            in_use: true,
        });
        self.textures.len() - 1
    }

    pub fn view(&self, index: usize) -> &wgpu::TextureView {
        &self.textures[index].view
    }

    pub fn texture(&self, index: usize) -> &wgpu::Texture {
        &self.textures[index].texture
    }
}

pub(crate) struct ReadbackRing {
    streams: HashMap<StreamKey, StreamSlots>,
}

struct StreamSlots {
    next: usize,
    slots: [ReadbackSlot; 3],
}

struct ReadbackSlot {
    buffer: Option<wgpu::Buffer>,
    buffer_size: u64,
    generation: u32,
    state: Arc<AtomicU8>,
}

const IDLE: u8 = 0;
const PENDING: u8 = 1;
const READY: u8 = 2;
const FAILED: u8 = 3;

impl ReadbackRing {
    pub fn new() -> Self {
        Self { streams: HashMap::new() }
    }

    pub fn schedule(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        view: ViewKey,
        output: OutputKind,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
    ) -> Option<ReadbackHandle> {
        let stream = StreamKey { view, output };
        let slots = self.streams.entry(stream).or_insert_with(StreamSlots::new);
        let slot_index = (0..slots.slots.len())
            .map(|offset| (slots.next + offset) % slots.slots.len())
            .find(|&index| slots.slots[index].state.load(Ordering::Acquire) == IDLE)?;
        slots.next = (slot_index + 1) % slots.slots.len();

        let bytes_per_row = width * 4;
        let padded_bytes_per_row = bytes_per_row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let required_size = u64::from(padded_bytes_per_row) * u64::from(height);
        let slot = &mut slots.slots[slot_index];
        if slot.buffer_size < required_size {
            slot.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sensor readback ring slot"),
                size: required_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }));
            slot.buffer_size = required_size;
        }
        slot.generation = slot.generation.wrapping_add(1);
        slot.state.store(PENDING, Ordering::Release);
        let buffer = slot.buffer.as_ref().expect("readback buffer was allocated");
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        Some(ReadbackHandle {
            stream,
            slot: slot_index as u8,
            generation: slot.generation,
            width,
            height,
            padded_bytes_per_row,
        })
    }

    pub fn begin_mapping(&self, handles: &[ReadbackHandle]) {
        for handle in handles {
            let slot = &self.streams[&handle.stream].slots[handle.slot as usize];
            let state = slot.state.clone();
            slot.buffer
                .as_ref()
                .expect("scheduled readback has a buffer")
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    state.store(if result.is_ok() { READY } else { FAILED }, Ordering::Release);
                });
        }
    }

    pub fn poll(
        &mut self,
        handle: ReadbackHandle,
    ) -> Result<Option<ReadbackData>, ()> {
        let Some(stream) = self.streams.get_mut(&handle.stream) else {
            return Err(());
        };
        let slot = &mut stream.slots[handle.slot as usize];
        if slot.generation != handle.generation {
            return Err(());
        }
        match slot.state.load(Ordering::Acquire) {
            PENDING => Ok(None),
            FAILED => {
                slot.state.store(IDLE, Ordering::Release);
                Err(())
            }
            READY => {
                let buffer = slot.buffer.as_ref().expect("ready readback has a buffer");
                let mapped = buffer.slice(..).get_mapped_range().map_err(|_| ())?;
                let row_bytes = handle.width as usize * 4;
                let mut bytes = Vec::with_capacity(row_bytes * handle.height as usize);
                for row in mapped.chunks_exact(handle.padded_bytes_per_row as usize) {
                    bytes.extend_from_slice(&row[..row_bytes]);
                }
                drop(mapped);
                buffer.unmap();
                slot.state.store(IDLE, Ordering::Release);
                Ok(Some(match handle.stream.output {
                    OutputKind::Color => ReadbackData::Color(bytes),
                    OutputKind::Depth => ReadbackData::Depth(
                        bytes.chunks_exact(4).map(|value| f32::from_le_bytes(value.try_into().unwrap())).collect(),
                    ),
                    OutputKind::ObjectId => ReadbackData::ObjectIds(
                        bytes.chunks_exact(4).map(|value| u32::from_le_bytes(value.try_into().unwrap())).collect(),
                    ),
                }))
            }
            _ => Err(()),
        }
    }
}

impl StreamSlots {
    fn new() -> Self {
        Self {
            next: 0,
            slots: std::array::from_fn(|_| ReadbackSlot {
                buffer: None,
                buffer_size: 0,
                generation: 0,
                state: Arc::new(AtomicU8::new(IDLE)),
            }),
        }
    }
}
