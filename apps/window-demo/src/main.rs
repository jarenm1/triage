use std::f32::consts::FRAC_PI_4;

use glam::{Mat4, Quat, Vec3};
use sim_graphics::{Camera, Frame, MeshData, MeshHandle, RenderPrimitive, Renderer, RendererError};
use sim_graphics_winit::Scene;

fn main() -> Result<(), sim_graphics_winit::WindowError> {
    sim_graphics_winit::run("Simulator Graphics", DemoScene::new())
}

struct DemoScene {
    frame: Frame,
    meshes: Option<(MeshHandle, MeshHandle)>,
}

impl DemoScene {
    fn new() -> Self {
        Self {
            frame: Frame::with_capacity(camera(0.0), 401),
            meshes: None,
        }
    }
}

impl Scene for DemoScene {
    fn initialize(&mut self, renderer: &mut Renderer) -> Result<(), RendererError> {
        self.meshes = Some((
            renderer.register_mesh(MeshData::cube())?,
            renderer.register_mesh(MeshData::plane())?,
        ));
        Ok(())
    }

    fn frame(&mut self, elapsed_seconds: f32) -> &Frame {
        let (cube_mesh, plane_mesh) = self
            .meshes
            .expect("the window adapter initializes scenes before rendering");
        self.frame.begin(camera(elapsed_seconds));
        self.frame.draw(RenderPrimitive {
            mesh: plane_mesh,
            transform: Mat4::from_scale_rotation_translation(
                Vec3::new(24.0, 1.0, 24.0),
                Quat::IDENTITY,
                Vec3::new(0.0, -0.5, 0.0),
            ),
            color: [0.18, 0.22, 0.26, 1.0],
        });
        for z in -10_i32..10 {
            for x in -10_i32..10 {
                self.frame.draw(RenderPrimitive {
                    mesh: cube_mesh,
                    transform: cube_transform(x, z, elapsed_seconds),
                    color: [
                        0.22 + (x + 10) as f32 / 40.0,
                        0.35 + (z + 10) as f32 / 50.0,
                        0.78,
                        1.0,
                    ],
                });
            }
        }
        &self.frame
    }
}

fn camera(elapsed_seconds: f32) -> Camera {
    let orbit = elapsed_seconds * 0.16;
    Camera {
        eye: Vec3::new(orbit.cos() * 20.5, 12.0, orbit.sin() * 20.5),
        target: Vec3::ZERO,
        up: Vec3::Y,
        vertical_fov_radians: FRAC_PI_4,
        near: 0.1,
        far: 100.0,
    }
}

fn cube_transform(x: i32, z: i32, elapsed_seconds: f32) -> Mat4 {
    let wave = (elapsed_seconds * 1.5 + x as f32 * 0.28 + z as f32 * 0.2).sin();
    let position = Vec3::new(x as f32 * 1.05, 0.1 + wave * 0.16, z as f32 * 1.05);
    let scale = Vec3::new(0.72, 0.8 + ((x + z).abs() % 4) as f32 * 0.22, 0.72);
    Mat4::from_scale_rotation_translation(scale, Quat::IDENTITY, position)
}
