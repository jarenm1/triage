use std::f32::consts::FRAC_PI_4;

use glam::{Mat4, Quat, Vec3};
use sim_graphics::{Camera, Cube};
use sim_graphics_winit::{Scene, SceneFrame};

fn main() -> Result<(), sim_graphics_winit::WindowError> {
    sim_graphics_winit::run("Simulator Graphics", DemoScene::new())
}

struct DemoScene {
    cubes: Vec<Cube>,
}

impl DemoScene {
    fn new() -> Self {
        let mut cubes = Vec::with_capacity(401);
        cubes.push(Cube {
            transform: Mat4::from_scale_rotation_translation(
                Vec3::new(24.0, 0.12, 24.0),
                Quat::IDENTITY,
                Vec3::new(0.0, -0.56, 0.0),
            ),
            color: [0.18, 0.22, 0.26, 1.0],
        });
        for z in -10_i32..10 {
            for x in -10_i32..10 {
                cubes.push(Cube {
                    transform: cube_transform(x, z, 0.0),
                    color: [
                        0.22 + (x + 10) as f32 / 40.0,
                        0.35 + (z + 10) as f32 / 50.0,
                        0.78,
                        1.0,
                    ],
                });
            }
        }
        Self { cubes }
    }
}

impl Scene for DemoScene {
    fn frame(&mut self, elapsed_seconds: f32) -> SceneFrame<'_> {
        for (index, cube) in self.cubes[1..].iter_mut().enumerate() {
            let x = index as i32 % 20 - 10;
            let z = index as i32 / 20 - 10;
            cube.transform = cube_transform(x, z, elapsed_seconds);
        }
        let orbit = elapsed_seconds * 0.16;
        SceneFrame {
            camera: Camera {
                eye: Vec3::new(orbit.cos() * 20.5, 12.0, orbit.sin() * 20.5),
                target: Vec3::ZERO,
                up: Vec3::Y,
                vertical_fov_radians: FRAC_PI_4,
                near: 0.1,
                far: 100.0,
            },
            cubes: &self.cubes,
        }
    }
}

fn cube_transform(x: i32, z: i32, elapsed_seconds: f32) -> Mat4 {
    let wave = (elapsed_seconds * 1.5 + x as f32 * 0.28 + z as f32 * 0.2).sin();
    let position = Vec3::new(x as f32 * 1.05, 0.1 + wave * 0.16, z as f32 * 1.05);
    let scale = Vec3::new(0.72, 0.8 + ((x + z).abs() % 4) as f32 * 0.22, 0.72);
    Mat4::from_scale_rotation_translation(scale, Quat::IDENTITY, position)
}
