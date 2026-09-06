use anyhow::{Context, Result, ensure};
use sim_graphics::{Frame, ReadbackData, RenderView, Renderer, ViewKey, ViewKind, ViewOutputs};
use sim_trajectory::{CameraMode, GROUND_OBJECT_ID, Recording, TrajectoryScene};
use std::{path::PathBuf, time::Instant};

pub async fn run() -> Result<()> {
    let mut args = std::env::args_os().skip(2);
    let input = PathBuf::from(
        args.next()
            .context("usage: render-smoke --trajectory FILE --output DIR")?,
    );
    ensure!(
        args.next().as_deref() == Some(std::ffi::OsStr::new("--output")),
        "expected --output DIR"
    );
    let output = PathBuf::from(args.next().context("missing output directory")?);
    ensure!(args.next().is_none(), "unexpected trajectory arguments");
    let recording = Recording::parse(&std::fs::read_to_string(input)?)?;
    std::fs::create_dir_all(&output)?;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let mut renderer = Renderer::new(&instance, None).await?;
    let mut scene = TrajectoryScene::new();
    scene.initialize(&mut renderer)?;
    let mut frame = Frame::with_capacity(1024, 1);
    let (width, height) = (960, 540);
    let mut indices = vec![
        0,
        recording.snapshots.len() / 2,
        recording.snapshots.len() - 1,
    ];
    indices.dedup();
    for index in indices {
        let snapshot = &recording.snapshots[index];
        for (name, mode) in [
            (
                "orbit",
                CameraMode::Orbit {
                    azimuth: 0.8,
                    elevation: 0.55,
                    distance: 9.0,
                },
            ),
            (
                "mounted",
                CameraMode::Mounted {
                    environment_id: recording.header.environment_ids[0],
                },
            ),
        ] {
            let prepare = Instant::now();
            frame.begin();
            frame.add_view(RenderView {
                key: ViewKey(1),
                kind: ViewKind::Sensor,
                camera: scene.camera(&recording.header, snapshot, mode),
                width,
                height,
                outputs: ViewOutputs::COLOR | ViewOutputs::DEPTH | ViewOutputs::OBJECT_ID,
            });
            scene.draw(&mut frame, snapshot);
            let prepare_us = prepare.elapsed().as_micros();
            let submit = Instant::now();
            renderer.execute_gpu(&frame, &[])?;
            let upload_render_submit_us = submit.elapsed().as_micros();
            let (done_tx, done_rx) = std::sync::mpsc::sync_channel(1);
            renderer.queue().on_submitted_work_done(move || {
                let _ = done_tx.send(());
            });
            while done_rx.try_recv().is_err() {
                renderer.device().poll(wgpu::PollType::Poll)?;
                std::thread::yield_now();
            }
            let gpu_completed_wall_us = submit.elapsed().as_micros();
            // GPU-completed wall time above includes CPU submission and polling:
            // it is not a GPU timestamp query. Readback is measured separately.
            let readback_start = Instant::now();
            let submission = renderer.execute(&frame, &[])?;
            let mut color = None;
            let mut depth = None;
            let mut object_ids = None;
            for handle in submission.readbacks {
                let data = loop {
                    if let Some(data) = renderer.poll_readback(handle)? {
                        break data;
                    }
                    std::thread::yield_now();
                };
                match data {
                    ReadbackData::Color(pixels) => color = Some(pixels),
                    ReadbackData::Depth(values) => depth = Some(values),
                    ReadbackData::ObjectIds(values) => object_ids = Some(values),
                }
            }
            let readback_wall_us = readback_start.elapsed().as_micros();
            let depth = depth.context("missing depth readback")?;
            let object_ids = object_ids.context("missing object ID readback")?;
            for (&id, &z) in object_ids.iter().zip(&depth) {
                ensure!(
                    z.is_finite() && z >= 0.0 && z <= recording.header.camera.far,
                    "invalid optical depth"
                );
                if id == 0 {
                    ensure!(
                        z == recording.header.camera.far,
                        "visible geometry incorrectly uses the background ID"
                    );
                } else {
                    ensure!(
                        id == GROUND_OBJECT_ID
                            || snapshot
                                .vehicles
                                .iter()
                                .any(|vehicle| id == vehicle.drone_object_id()
                                    || id == vehicle.target_object_id()),
                        "rendered object ID has no trajectory identity"
                    );
                }
            }
            let path = output.join(format!("step-{:010}-{name}.png", snapshot.step));
            image::save_buffer(
                &path,
                &color.context("missing color readback")?,
                width,
                height,
                image::ColorType::Rgba8,
            )?;
            println!(
                "trajectory step={} time={:.3} camera={} primitives={} CPU_prepare_us={} CPU_upload_render_submit_us={} GPU_completed_wall_us={} render_readback_wall_us={} output={}",
                snapshot.step,
                snapshot.time_seconds,
                name,
                frame.primitives().len(),
                prepare_us,
                upload_render_submit_us,
                gpu_completed_wall_us,
                readback_wall_us,
                path.display()
            );
        }
    }
    Ok(())
}
