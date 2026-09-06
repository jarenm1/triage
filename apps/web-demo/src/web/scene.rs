use glam::{Mat4, Vec3};
use sim_graphics::{MeshData, MeshHandle, RenderPrimitive, Renderer};

pub struct Object {
    pub id: u32,
    pub name: &'static str,
}

pub struct Scene {
    pub name: &'static str,
    pub target: Vec3,
    pub objects: Vec<Object>,
    pub primitives: Vec<RenderPrimitive>,
}

impl Scene {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            target: Vec3::new(0.0, 1.0, 0.0),
            objects: Vec::new(),
            primitives: Vec::new(),
        }
    }

    fn object(&mut self, id: u32, name: &'static str) {
        self.objects.push(Object { id, name });
    }

    fn part(
        &mut self,
        mesh: MeshHandle,
        id: u32,
        position: [f32; 3],
        scale: [f32; 3],
        color: [f32; 3],
    ) {
        self.primitives.push(RenderPrimitive {
            mesh,
            object_id: id,
            transform: Mat4::from_scale_rotation_translation(
                scale.into(),
                glam::Quat::IDENTITY,
                position.into(),
            ),
            color: [color[0], color[1], color[2], 1.0],
        });
    }

    fn gate(
        &mut self,
        cube: MeshHandle,
        id: u32,
        name: &'static str,
        x: f32,
        z: f32,
        color: [f32; 3],
    ) {
        self.object(id, name);
        for dx in [-1.65, 1.65] {
            self.part(cube, id, [x + dx, 1.6, z], [0.35, 3.2, 0.45], color);
        }
        self.part(cube, id, [x, 3.15, z], [3.65, 0.35, 0.45], color);
    }
}

pub fn build(renderer: &mut Renderer) -> Result<[Scene; 2], sim_graphics::RendererError> {
    let plane = renderer.register_mesh(MeshData::plane())?;
    let cube = renderer.register_mesh(MeshData::cube())?;
    let cylinder = renderer.register_mesh(MeshData::cylinder(32))?;
    let sphere = renderer.register_mesh(MeshData::sphere(20, 32))?;

    let mut courtyard = Scene::new("Occlusion courtyard");
    courtyard.object(1, "Courtyard floor");
    courtyard.part(
        plane,
        1,
        [0.0, 0.0, 0.0],
        [14.0, 1.0, 12.0],
        [0.22, 0.29, 0.32],
    );
    courtyard.object(2, "Rear boundary");
    courtyard.part(
        cube,
        2,
        [0.0, 0.65, -5.4],
        [13.6, 1.3, 0.3],
        [0.43, 0.48, 0.48],
    );
    courtyard.object(3, "Left boundary");
    courtyard.part(
        cube,
        3,
        [-6.7, 0.65, 0.0],
        [0.3, 1.3, 11.0],
        [0.43, 0.48, 0.48],
    );
    courtyard.gate(
        cube,
        101,
        "Amber gate (three parts)",
        -2.25,
        1.4,
        [0.95, 0.44, 0.055],
    );
    courtyard.gate(
        cube,
        202,
        "Teal gate (three parts)",
        2.0,
        -2.1,
        [0.04, 0.62, 0.54],
    );
    courtyard.object(301, "Foreground occluder");
    courtyard.part(
        cube,
        301,
        [2.7, 1.1, 2.8],
        [2.0, 2.2, 1.35],
        [0.68, 0.13, 0.18],
    );
    courtyard.object(302, "Partly hidden sphere");
    courtyard.part(
        sphere,
        302,
        [2.45, 1.0, 0.7],
        [1.8, 1.8, 1.8],
        [0.25, 0.36, 0.9],
    );
    courtyard.object(303, "Tall cylinder");
    courtyard.part(
        cylinder,
        303,
        [-4.4, 1.0, -2.6],
        [1.15, 2.0, 1.15],
        [0.56, 0.19, 0.74],
    );
    courtyard.object(304, "Low cylinder");
    courtyard.part(
        cylinder,
        304,
        [4.8, 0.55, -3.5],
        [1.5, 1.1, 1.5],
        [0.7, 0.65, 0.11],
    );
    courtyard.object(65_537, "Small blue cube");
    courtyard.part(
        cube,
        65_537,
        [-0.4, 0.45, -3.8],
        [0.9, 0.9, 0.9],
        [0.12, 0.43, 0.8],
    );
    courtyard.object(4_000_000_001, "High-ID marker sphere");
    courtyard.part(
        sphere,
        4_000_000_001,
        [-4.5, 0.55, 3.8],
        [1.1, 1.1, 1.1],
        [0.8, 0.65, 0.35],
    );

    let mut calibration = Scene::new("Calibration cubes");
    calibration.object(1, "Calibration floor");
    calibration.part(
        plane,
        1,
        [0.0, 0.0, 0.0],
        [12.0, 1.0, 10.0],
        [0.23, 0.27, 0.31],
    );
    for (id, name, position, scale, color) in [
        (
            11,
            "One metre cube",
            [-3.0, 0.5, 2.0],
            [1.0, 1.0, 1.0],
            [0.85, 0.17, 0.12],
        ),
        (
            12,
            "Two metre cube",
            [0.0, 1.0, 0.0],
            [2.0, 2.0, 2.0],
            [0.08, 0.62, 0.37],
        ),
        (
            65_537,
            "Three metre cube",
            [3.1, 1.5, -2.6],
            [3.0, 3.0, 3.0],
            [0.13, 0.33, 0.85],
        ),
        (
            4_000_000_001,
            "Rear occluded cube",
            [-1.2, 0.6, -3.4],
            [1.2, 1.2, 1.2],
            [0.87, 0.56, 0.06],
        ),
    ] {
        calibration.object(id, name);
        calibration.part(cube, id, position, scale, color);
    }
    Ok([courtyard, calibration])
}

pub fn palette(id: u32) -> String {
    if id == 0 {
        return "#000000".into();
    }
    let mut hash = id;
    hash = (hash ^ (hash >> 16)).wrapping_mul(0x7feb352d);
    hash = (hash ^ (hash >> 15)).wrapping_mul(0x846ca68b);
    hash ^= hash >> 16;
    let channel = |shift: u32| 64 + ((hash >> shift) & 255u32) * 191 / 255;
    format!("#{:02x}{:02x}{:02x}", channel(0), channel(8), channel(16))
}
