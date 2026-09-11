use std::path::PathBuf;

use anyhow::{Context, Result, bail, ensure};
use sim_inspection::Inspector;
use sim_scene::SceneConfig;

pub async fn run() -> Result<()> {
    let mut args = std::env::args_os().skip(2);
    let mut scene_path = None;
    let mut output = PathBuf::from("target/inspection");
    let mut verify = false;
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--scene") => {
                scene_path = Some(PathBuf::from(
                    args.next().context("--scene requires a path")?,
                ));
            }
            Some("--output") => {
                output = PathBuf::from(args.next().context("--output requires a directory")?);
            }
            Some("--verify") => verify = true,
            Some("--help") => {
                println!(
                    "render-smoke --inspect [--scene scene.json] [--output directory] [--verify]"
                );
                return Ok(());
            }
            _ => bail!("unknown inspection argument: {}", arg.to_string_lossy()),
        }
    }
    let config = match scene_path {
        Some(path) => SceneConfig::load(path)?,
        None => SceneConfig::default(),
    };
    config.validate()?;
    let mut inspector = Inspector::new().await?;
    let capture = inspector.capture(&config)?;
    capture.save(&config, &output)?;
    println!(
        "Saved synchronized inspection outputs to {}",
        output.display()
    );
    if verify {
        verify_geometry(&mut inspector)?;
        // Reload the exact saved scene and exercise the same renderer again.
        let path = output.join("replay-scene.json");
        config.save(&path)?;
        let replay = inspector.capture(&SceneConfig::load(&path)?)?;
        ensure!(capture.color == replay.color, "RGB changed on scene replay");
        ensure!(
            capture.depth == replay.depth,
            "depth changed on scene replay"
        );
        ensure!(
            capture.object_ids == replay.object_ids,
            "instance IDs changed on scene replay"
        );
        println!("Scene save/reload: exact RGB, depth and instance-ID replay passed");
    }
    Ok(())
}

fn verify_geometry(inspector: &mut Inspector) -> Result<()> {
    let mut config = SceneConfig::default();
    config.sensor.eye = [0.0, 0.0, 5.0];
    config.sensor.target = [0.0, 0.0, 0.0];
    config.sensor.up = [0.0, 1.0, 0.0];
    config.sensor.vertical_fov_radians = std::f32::consts::FRAC_PI_2;
    config.sensor.width = 320;
    config.sensor.height = 240;
    config.sensor.near = 0.1;
    config.sensor.far = 30.0;
    ensure!(
        config.objects.len() >= 2,
        "verification needs two scene objects"
    );
    config.objects.truncate(2);
    config.objects[0].parts[0].position = [0.4, 0.3, 1.0];
    config.objects[0].parts[0].scale = [1.0, 1.0, 1.0];
    config.objects[1].parts[0].position = [0.0, 0.0, -1.0];
    config.objects[1].parts[0].scale = [3.0, 3.0, 1.0];
    let front_id = config.objects[0].id;
    let rear_id = config.objects[1].id;
    let capture = inspector.capture(&config)?;
    let mut counts = [0usize; 3];
    for y in 0..240 {
        for x in 0..320 {
            // Pixel centers; optical x right, y down, z forward. The scene is Y-up.
            let dx = (x as f32 + 0.5 - 160.0) / 120.0;
            let dy = -(y as f32 + 0.5 - 120.0) / 120.0;
            let front = (-0.1..=0.9).contains(&(dx * 3.5)) && (-0.2..=0.8).contains(&(dy * 3.5));
            let rear = (dx * 5.5).abs() <= 1.5 && (dy * 5.5).abs() <= 1.5;
            let (id, depth, category) = if front {
                (front_id, 3.5, 0)
            } else if rear {
                (rear_id, 5.5, 1)
            } else {
                (0, config.sensor.far, 2)
            };
            let index = y * 320 + x;
            ensure!(
                capture.object_ids[index] == id,
                "projection/occlusion mismatch at ({x},{y}): expected ID {id}, got {}",
                capture.object_ids[index]
            );
            ensure!(
                (capture.depth[index] - depth).abs() < 0.0001,
                "optical depth mismatch at ({x},{y}): expected {depth}, got {}",
                capture.depth[index]
            );
            counts[category] += 1;
        }
    }
    ensure!(
        counts.iter().all(|count| *count > 0),
        "fixture did not exercise every surface"
    );
    println!(
        "Analytical projection/depth/occlusion: all 76,800 pixels passed; front={}, rear={}, background={}",
        counts[0], counts[1], counts[2]
    );
    Ok(())
}
