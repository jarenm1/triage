use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

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

    /// Heightfield terrain from a noise function. `size` is the grid
    /// resolution (verts per side), `extent` the world span, `height`
    /// the max elevation. `noise(x, z) -> [0,1]` supplies the heightmap.
    pub fn heightfield(
        size: u32,
        extent: f32,
        height: f32,
        noise: impl Fn(f32, f32) -> f32,
    ) -> Self {
        let size = size.max(2);
        let n = size as usize;
        let mut vertices = Vec::with_capacity(n * n);
        let step = extent / (size - 1) as f32;
        let half = extent * 0.5;
        let mut heights = vec![0.0f32; n * n];
        for iz in 0..n {
            for ix in 0..n {
                let x = ix as f32 * step - half;
                let z = iz as f32 * step - half;
                heights[iz * n + ix] = noise(x, z) * height;
            }
        }
        for iz in 0..n {
            for ix in 0..n {
                let x = ix as f32 * step - half;
                let z = iz as f32 * step - half;
                let y = heights[iz * n + ix];
                // central-difference normal
                let hl = heights[iz * n + ix.saturating_sub(1)];
                let hr = heights[iz * n + (ix + 1).min(n - 1)];
                let hd = heights[iz.saturating_sub(1) * n + ix];
                let hu = heights[(iz + 1).min(n - 1) * n + ix];
                let normal = Vec3::new(hl - hr, 2.0 * step, hd - hu).normalize();
                vertices.push(MeshVertex {
                    position: [x, y, z],
                    normal: normal.to_array(),
                });
            }
        }
        let mut indices = Vec::with_capacity((n - 1) * (n - 1) * 6);
        for iz in 0..(n - 1) {
            for ix in 0..(n - 1) {
                let a = (iz * n + ix) as u32;
                let b = a + 1;
                let c = a + n as u32;
                let d = c + 1;
                indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
        Self { vertices, indices }
    }

    /// Flat subdivided grid for vertex-shader-displaced terrain.
    /// `size` verts per side, `extent` world span. Normals are up;
    /// the shader displaces Y by world-space noise.
    pub fn grid(size: u32, extent: f32) -> Self {
        let size = size.max(2);
        let n = size as usize;
        let mut vertices = Vec::with_capacity(n * n);
        let step = extent / (size - 1) as f32;
        let half = extent * 0.5;
        for iz in 0..n {
            for ix in 0..n {
                vertices.push(MeshVertex {
                    position: [ix as f32 * step - half, 0.0, iz as f32 * step - half],
                    normal: [0.0, 1.0, 0.0],
                });
            }
        }
        let mut indices = Vec::with_capacity((n - 1) * (n - 1) * 6);
        for iz in 0..(n - 1) {
            for ix in 0..(n - 1) {
                let a = (iz * n + ix) as u32;
                let b = a + 1;
                let c = a + n as u32;
                let d = c + 1;
                indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
        Self { vertices, indices }
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
    /// Procedural texture: (kind, scale, seed, contrast).
    /// kind 0=none, 1=checker, 2=stripes, 3=value-noise, 4=gradient,
    /// 5=voronoi, 6=brick, 7=grid, 8=rings, 9=terrain-displace.
    pub pattern: [f32; 4],
    /// Aux: (snap_flag, base_x, base_z, unused). snap_flag > 0.5
    /// displaces the object by terrain_height(base_xz).
    pub aux: [f32; 4],
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
