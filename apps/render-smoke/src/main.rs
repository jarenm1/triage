use std::{f32::consts::FRAC_PI_4, path::PathBuf};

use anyhow::{Context, Result};
use glam::{Mat4, Quat, Vec3};
use sim_graphics::{Camera, Cube, OffscreenTarget, Renderer};

fn main() -> Result<()> {
    pollster::block_on(run())
}

async fn run() -> Result<()> {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/render-smoke.png"));
    let (width, height) = (960, 540);
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let mut renderer = Renderer::new(&instance, None).await?;
    let target = OffscreenTarget::new(&renderer, width, height);

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
            let position = Vec3::new(
                x as f32 * 1.05,
                0.05 + ((x * z).abs() % 5) as f32 * 0.16,
                z as f32 * 1.05,
            );
            let scale = Vec3::new(0.72, 0.7 + ((x + z).abs() % 4) as f32 * 0.22, 0.72);
            let color = [
                0.22 + (x + 10) as f32 / 40.0,
                0.35 + (z + 10) as f32 / 50.0,
                0.78,
                1.0,
            ];
            cubes.push(Cube {
                transform: Mat4::from_scale_rotation_translation(scale, Quat::IDENTITY, position),
                color,
            });
        }
    }

    let camera = Camera {
        eye: Vec3::new(13.5, 12.0, 16.0),
        target: Vec3::new(0.0, 0.0, 0.0),
        up: Vec3::Y,
        vertical_fov_radians: FRAC_PI_4,
        near: 0.1,
        far: 100.0,
    };
    renderer.render(target.render_target(), camera, &cubes)?;
    let pixels = target.read_rgba(&renderer).await?;
    image::save_buffer(&output, &pixels, width, height, image::ColorType::Rgba8)
        .with_context(|| format!("saving {}", output.display()))?;
    println!("rendered {} instances to {}", cubes.len(), output.display());
    Ok(())
}
