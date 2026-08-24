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

    pub fn cylinder(segments: u32) -> Self {
        let segments = segments.max(3);
        let mut vertices = Vec::with_capacity((segments * 4 + 2) as usize);
        let mut indices = Vec::with_capacity((segments * 12) as usize);
        let step = std::f32::consts::TAU / segments as f32;

        // Side vertices
        let side_start = vertices.len() as u32;
        for i in 0..=segments {
            let angle = i as f32 * step;
            let cos = angle.cos();
            let sin = angle.sin();
            let normal = [cos, 0.0, sin];
            // Bottom vertex
            vertices.push(MeshVertex {
                position: [cos * 0.5, -0.5, sin * 0.5],
                normal,
            });
            // Top vertex
            vertices.push(MeshVertex {
                position: [cos * 0.5, 0.5, sin * 0.5],
                normal,
            });
        }
        for i in 0..segments {
            let v00 = side_start + i * 2;
            let v01 = side_start + i * 2 + 1;
            let v10 = side_start + (i + 1) * 2;
            let v11 = side_start + (i + 1) * 2 + 1;
            indices.extend_from_slice(&[v00, v01, v10, v10, v01, v11]);
        }

        // Top cap (normal [0, 1, 0])
        let top_center_idx = vertices.len() as u32;
        vertices.push(MeshVertex {
            position: [0.0, 0.5, 0.0],
            normal: [0.0, 1.0, 0.0],
        });
        let top_ring_start = vertices.len() as u32;
        for i in 0..segments {
            let angle = i as f32 * step;
            vertices.push(MeshVertex {
                position: [angle.cos() * 0.5, 0.5, angle.sin() * 0.5],
                normal: [0.0, 1.0, 0.0],
            });
        }
        for i in 0..segments {
            let next = (i + 1) % segments;
            indices.extend_from_slice(&[
                top_center_idx,
                top_ring_start + next,
                top_ring_start + i,
            ]);
        }

        // Bottom cap (normal [0, -1, 0])
        let bottom_center_idx = vertices.len() as u32;
        vertices.push(MeshVertex {
            position: [0.0, -0.5, 0.0],
            normal: [0.0, -1.0, 0.0],
        });
        let bottom_ring_start = vertices.len() as u32;
        for i in 0..segments {
            let angle = i as f32 * step;
            vertices.push(MeshVertex {
                position: [angle.cos() * 0.5, -0.5, angle.sin() * 0.5],
                normal: [0.0, -1.0, 0.0],
            });
        }
        for i in 0..segments {
            let next = (i + 1) % segments;
            indices.extend_from_slice(&[
                bottom_center_idx,
                bottom_ring_start + i,
                bottom_ring_start + next,
            ]);
        }

        Self { vertices, indices }
    }

    pub fn sphere(lat_segments: u32, lon_segments: u32) -> Self {
        let lat_segments = lat_segments.max(3);
        let lon_segments = lon_segments.max(3);
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for lat in 0..=lat_segments {
            let theta = lat as f32 * std::f32::consts::PI / lat_segments as f32;
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();

            for lon in 0..=lon_segments {
                let phi = lon as f32 * std::f32::consts::TAU / lon_segments as f32;
                let sin_phi = phi.sin();
                let cos_phi = phi.cos();

                let x = cos_phi * sin_theta;
                let y = cos_theta;
                let z = sin_phi * sin_theta;

                vertices.push(MeshVertex {
                    position: [x * 0.5, y * 0.5, z * 0.5],
                    normal: [x, y, z],
                });
            }
        }

        for lat in 0..lat_segments {
            for lon in 0..lon_segments {
                let first = lat * (lon_segments + 1) + lon;
                let second = first + lon_segments + 1;

                indices.extend_from_slice(&[first, first + 1, second]);
                indices.extend_from_slice(&[second, first + 1, second + 1]);
            }
        }

        Self { vertices, indices }
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

    #[test]
    fn cylinder_triangles_wind_counter_clockwise() {
        let mesh = MeshData::cylinder(16);
        for triangle in mesh.indices.chunks_exact(3) {
            let a = Vec3::from_array(mesh.vertices[triangle[0] as usize].position);
            let b = Vec3::from_array(mesh.vertices[triangle[1] as usize].position);
            let c = Vec3::from_array(mesh.vertices[triangle[2] as usize].position);
            let n0 = Vec3::from_array(mesh.vertices[triangle[0] as usize].normal);
            let cross = (b - a).cross(c - a);
            if cross.length_squared() > 1e-6 {
                assert!(
                    cross.dot(n0) >= 0.0,
                    "cylinder triangle faces into the mesh"
                );
            }
        }
    }

    #[test]
    fn sphere_triangles_wind_counter_clockwise() {
        let mesh = MeshData::sphere(12, 16);
        for triangle in mesh.indices.chunks_exact(3) {
            let a = Vec3::from_array(mesh.vertices[triangle[0] as usize].position);
            let b = Vec3::from_array(mesh.vertices[triangle[1] as usize].position);
            let c = Vec3::from_array(mesh.vertices[triangle[2] as usize].position);
            let n0 = Vec3::from_array(mesh.vertices[triangle[0] as usize].normal);
            let cross = (b - a).cross(c - a);
            if cross.length_squared() > 1e-6 {
                assert!(
                    cross.dot(n0) >= 0.0,
                    "sphere triangle faces into the mesh"
                );
            }
        }
    }
}
