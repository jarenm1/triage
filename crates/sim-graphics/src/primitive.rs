use bytemuck::{Pod, Zeroable};
use glam::Mat4;

use crate::resource::Handle;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

pub struct MeshData {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
}

impl MeshData {
    pub fn cube() -> Self {
        let mut vertices = Vec::with_capacity(24);
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
        for (normal, positions) in faces {
            vertices.extend(
                positions
                    .into_iter()
                    .map(|position| MeshVertex { position, normal }),
            );
        }

        let mut indices = Vec::with_capacity(36);
        for face in 0..6_u32 {
            let base = face * 4;
            indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
        }
        Self { vertices, indices }
    }

    pub fn plane() -> Self {
        Self {
            vertices: vec![
                MeshVertex {
                    position: [-0.5, 0.0, -0.5],
                    normal: [0.0, 1.0, 0.0],
                },
                MeshVertex {
                    position: [0.5, 0.0, -0.5],
                    normal: [0.0, 1.0, 0.0],
                },
                MeshVertex {
                    position: [0.5, 0.0, 0.5],
                    normal: [0.0, 1.0, 0.0],
                },
                MeshVertex {
                    position: [-0.5, 0.0, 0.5],
                    normal: [0.0, 1.0, 0.0],
                },
            ],
            indices: vec![0, 2, 1, 0, 3, 2],
        }
    }
}

pub struct Mesh {
    pub(crate) vertices: wgpu::Buffer,
    pub(crate) indices: wgpu::Buffer,
    pub(crate) index_count: u32,
}

pub type MeshHandle = Handle<Mesh>;

#[derive(Clone, Copy, Debug)]
pub struct RenderPrimitive {
    pub mesh: MeshHandle,
    pub transform: Mat4,
    pub color: [f32; 4],
    pub object_id: u32,
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::MeshData;

    #[test]
    fn cube_triangles_wind_counter_clockwise_from_outside() {
        let mesh = MeshData::cube();
        for triangle in mesh.indices.chunks_exact(3) {
            let a = Vec3::from_array(mesh.vertices[triangle[0] as usize].position);
            let b = Vec3::from_array(mesh.vertices[triangle[1] as usize].position);
            let c = Vec3::from_array(mesh.vertices[triangle[2] as usize].position);
            let expected_normal = Vec3::from_array(mesh.vertices[triangle[0] as usize].normal);

            assert!(
                (b - a).cross(c - a).dot(expected_normal) > 0.0,
                "triangle {triangle:?} faces into the mesh"
            );
        }
    }
}
