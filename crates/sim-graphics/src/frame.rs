use crate::{Camera, RenderPrimitive};

pub struct Frame {
    camera: Camera,
    primitives: Vec<RenderPrimitive>,
}

impl Frame {
    pub fn new(camera: Camera) -> Self {
        Self {
            camera,
            primitives: Vec::new(),
        }
    }

    pub fn with_capacity(camera: Camera, primitive_capacity: usize) -> Self {
        Self {
            camera,
            primitives: Vec::with_capacity(primitive_capacity),
        }
    }

    pub fn begin(&mut self, camera: Camera) -> &mut Self {
        self.camera = camera;
        self.primitives.clear();
        self
    }

    pub fn draw(&mut self, primitive: RenderPrimitive) -> &mut Self {
        self.primitives.push(primitive);
        self
    }

    pub fn camera(&self) -> Camera {
        self.camera
    }

    pub fn primitives(&self) -> &[RenderPrimitive] {
        &self.primitives
    }

    pub fn capacity(&self) -> usize {
        self.primitives.capacity()
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::{Camera, Frame};

    fn camera() -> Camera {
        Camera {
            eye: Vec3::Z,
            target: Vec3::ZERO,
            up: Vec3::Y,
            vertical_fov_radians: 1.0,
            near: 0.1,
            far: 10.0,
        }
    }

    #[test]
    fn beginning_a_frame_reuses_command_capacity() {
        let mut frame = Frame::with_capacity(camera(), 128);
        let capacity = frame.capacity();
        frame.begin(camera());
        assert_eq!(frame.capacity(), capacity);
        assert!(frame.primitives().is_empty());
    }
}
