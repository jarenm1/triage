use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail, ensure};
use glam::{Mat4, Quat, Vec3};
use serde::Serialize;
use sim_graphics::{
    Camera, Frame, MeshData, ReadbackData, RenderPrimitive, RenderView, Renderer, ViewKey,
    ViewKind, ViewOutputs,
};

const SPEC: &str = "E000/2";
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

#[derive(Debug)]
struct Args {
    envs: usize,
    steps: usize,
    warmup: usize,
    width: u32,
    height: u32,
    output: PathBuf,
}

#[derive(Serialize)]
struct Manifest {
    experiment: &'static str,
    source_revision: String,
    envs: usize,
    steps: usize,
    warmup: usize,
    width: u32,
    height: u32,
    color_format: &'static str,
    renderer: String,
    started_unix_ns: u128,
}

#[derive(Serialize)]
struct StepMetrics {
    step: usize,
    envs: usize,
    rendered_frames: usize,
    rendered_bytes: usize,
    checksum_fnv1a64: String,
    submit_ms: f64,
    readback_ms: f64,
    total_ms: f64,
}

fn main() -> Result<()> {
    pollster::block_on(run())
}

async fn run() -> Result<()> {
    let args = Args::parse()?;
    prepare_output(&args.output)?;
    let started = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let mut renderer = Renderer::new(&instance, None).await?;
    let adapter_info = renderer.adapter().get_info();
    let renderer_name = format!(
        "{}; backend={:?}; device_type={:?}; driver={}",
        adapter_info.name, adapter_info.backend, adapter_info.device_type, adapter_info.driver
    );

    let manifest = Manifest {
        experiment: SPEC,
        source_revision: source_revision(),
        envs: args.envs,
        steps: args.steps,
        warmup: args.warmup,
        width: args.width,
        height: args.height,
        color_format: "RGBA8_UNORM_SENSOR",
        renderer: renderer_name,
        started_unix_ns: started,
    };
    write_json(&args.output.join("manifest.json"), &manifest)?;

    let cube = renderer.register_mesh(MeshData::cube())?;
    let plane = renderer.register_mesh(MeshData::plane())?;
    let mut frame = Frame::with_capacity(args.envs * 4, args.envs);
    let mut metrics = Vec::with_capacity(args.steps);
    let mut sample_saved = false;

    for step in 0..(args.warmup + args.steps) {
        let measured = step >= args.warmup;
        let total_start = Instant::now();
        frame.begin();
        populate_frame(
            &mut frame,
            args.envs,
            args.width,
            args.height,
            cube,
            plane,
            step,
        );

        let submit_start = Instant::now();
        let submission = renderer.execute(&frame, &[])?;
        let submit_ms = submit_start.elapsed().as_secs_f64() * 1_000.0;

        let readback_start = Instant::now();
        let mut checksum = FNV_OFFSET;
        let mut rendered_bytes = 0usize;
        let mut sample = None;
        for (env, handle) in submission.readbacks.into_iter().enumerate() {
            let data = loop {
                if let Some(data) = renderer.poll_readback(handle)? {
                    break data;
                }
                std::thread::yield_now();
            };
            let pixels = match data {
                ReadbackData::Color(pixels) => pixels,
                other => bail!("expected color readback, received {other:?}"),
            };
            rendered_bytes += pixels.len();
            for byte in &pixels {
                checksum ^= u64::from(*byte);
                checksum = checksum.wrapping_mul(FNV_PRIME);
            }
            if measured && env == 0 && !sample_saved {
                sample = Some(pixels);
            }
        }
        let readback_ms = readback_start.elapsed().as_secs_f64() * 1_000.0;

        if measured {
            if let Some(pixels) = sample {
                image::save_buffer(
                    args.output.join("sample-env-000.png"),
                    &pixels,
                    args.width,
                    args.height,
                    image::ColorType::Rgba8,
                )?;
                sample_saved = true;
            }
            metrics.push(StepMetrics {
                step: step - args.warmup,
                envs: args.envs,
                rendered_frames: args.envs,
                rendered_bytes,
                checksum_fnv1a64: format!("{checksum:016x}"),
                submit_ms,
                readback_ms,
                total_ms: total_start.elapsed().as_secs_f64() * 1_000.0,
            });
        }
    }

    let mut jsonl = String::new();
    for metric in &metrics {
        jsonl.push_str(&serde_json::to_string(metric)?);
        jsonl.push('\n');
    }
    fs::write(args.output.join("metrics.jsonl"), jsonl)?;
    write_json(
        &args.output.join("summary.json"),
        &Summary::from_metrics(&metrics, args.envs, args.width, args.height),
    )?;
    println!("E000 RGB benchmark complete: {}", args.output.display());
    Ok(())
}

#[derive(Serialize)]
struct Summary {
    experiment: &'static str,
    measured_steps: usize,
    envs: usize,
    width: u32,
    height: u32,
    mean_submit_ms: f64,
    mean_readback_ms: f64,
    mean_total_ms: f64,
    frames_per_second: f64,
    pixels_per_second: f64,
    last_checksum_fnv1a64: Option<String>,
}

impl Summary {
    fn from_metrics(metrics: &[StepMetrics], envs: usize, width: u32, height: u32) -> Self {
        let count = metrics.len() as f64;
        let mean_submit_ms = metrics.iter().map(|m| m.submit_ms).sum::<f64>() / count.max(1.0);
        let mean_readback_ms = metrics.iter().map(|m| m.readback_ms).sum::<f64>() / count.max(1.0);
        let mean_total_ms = metrics.iter().map(|m| m.total_ms).sum::<f64>() / count.max(1.0);
        let frames_per_second = if mean_total_ms > 0.0 {
            envs as f64 * 1_000.0 / mean_total_ms
        } else {
            0.0
        };
        Self {
            experiment: SPEC,
            measured_steps: metrics.len(),
            envs,
            width,
            height,
            mean_submit_ms,
            mean_readback_ms,
            mean_total_ms,
            frames_per_second,
            pixels_per_second: frames_per_second * width as f64 * height as f64,
            last_checksum_fnv1a64: metrics.last().map(|m| m.checksum_fnv1a64.clone()),
        }
    }
}

impl Args {
    fn parse() -> Result<Self> {
        let mut envs = 256usize;
        let mut steps = 10usize;
        let mut warmup = 2usize;
        let mut width = 64u32;
        let mut height = 64u32;
        let mut output = None;
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--envs" => envs = next_number(&mut args, &flag)?,
                "--steps" => steps = next_number(&mut args, &flag)?,
                "--warmup" => warmup = next_number(&mut args, &flag)?,
                "--width" => width = next_number(&mut args, &flag)?,
                "--height" => height = next_number(&mut args, &flag)?,
                "--output" => {
                    output = Some(PathBuf::from(
                        args.next().context("--output requires a path")?,
                    ))
                }
                "--help" | "-h" => {
                    println!(
                        "rgb-benchmark [--envs N] [--steps N] [--warmup N] [--width N] [--height N] [--output DIR]"
                    );
                    return Err(anyhow::anyhow!("help requested"));
                }
                _ => bail!("unknown argument {flag}"),
            }
        }
        ensure!(envs > 0 && steps > 0, "--envs and --steps must be positive");
        ensure!(
            width > 0 && height > 0,
            "--width and --height must be positive"
        );
        let output = output.unwrap_or_else(|| {
            PathBuf::from(format!(
                "target/experiments/e000-rgb-benchmark-{}",
                unique_stamp()
            ))
        });
        Ok(Self {
            envs,
            steps,
            warmup,
            width,
            height,
            output,
        })
    }
}

fn next_number<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    args.next()
        .with_context(|| format!("{flag} requires a value"))?
        .parse()
        .map_err(|error| anyhow::anyhow!("invalid {flag} value: {error}"))
}

fn prepare_output(path: &Path) -> Result<()> {
    if path.exists() {
        bail!("output directory already exists: {}", path.display());
    }
    fs::create_dir_all(path).with_context(|| format!("creating {}", path.display()))?;
    Ok(())
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn unique_stamp() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_nanos()
}

fn source_revision() -> String {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|revision| !revision.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

fn populate_frame(
    frame: &mut Frame,
    envs: usize,
    width: u32,
    height: u32,
    cube: sim_graphics::MeshHandle,
    plane: sim_graphics::MeshHandle,
    step: usize,
) {
    let grid = (envs as f32).sqrt().ceil() as usize;
    let spacing = 16.0f32;
    let fov = 60.0f32.to_radians();
    for env in 0..envs {
        let gx = env % grid;
        let gz = env / grid;
        let origin = Vec3::new(gx as f32 * spacing, 0.0, gz as f32 * spacing);
        frame.add_view(RenderView {
            key: ViewKey(env as u64),
            kind: ViewKind::Sensor,
            camera: Camera {
                eye: origin + Vec3::new(0.0, 2.0, 4.5),
                target: origin + Vec3::new(0.0, 1.0, -2.0),
                up: Vec3::Y,
                vertical_fov_radians: fov,
                near: 0.05,
                far: 12.0,
            },
            width,
            height,
            outputs: ViewOutputs::COLOR,
        });
        let phase = step as f32 * 0.035 + env as f32 * 0.017;
        let drift = phase.sin() * 0.35;
        frame.draw(RenderPrimitive {
            mesh: plane,
            transform: Mat4::from_scale_rotation_translation(
                Vec3::new(9.0, 1.0, 13.0),
                Quat::IDENTITY,
                origin + Vec3::new(0.0, -0.05, -2.5),
            ),
            color: [0.16, 0.19, 0.22, 1.0],
            object_id: (env as u32) * 10 + 1,
        });
        let color = palette(env);
        for (index, (position, scale)) in [
            (Vec3::new(-1.4 + drift, 0.8, -2.4), Vec3::new(0.9, 1.6, 0.8)),
            (Vec3::new(1.1, 1.1, -4.2), Vec3::new(1.5, 2.2, 0.9)),
            (
                Vec3::new(-0.1, 0.55, -6.1 - drift),
                Vec3::new(2.5, 1.1, 0.7),
            ),
        ]
        .into_iter()
        .enumerate()
        {
            frame.draw(RenderPrimitive {
                mesh: cube,
                transform: Mat4::from_scale_rotation_translation(
                    scale,
                    Quat::IDENTITY,
                    origin + position,
                ),
                color: [
                    (color[0] + index as f32 * 0.07).min(1.0),
                    (color[1] + index as f32 * 0.05).min(1.0),
                    (color[2] + index as f32 * 0.03).min(1.0),
                    1.0,
                ],
                object_id: (env as u32) * 10 + 2 + index as u32,
            });
        }
    }
}

fn palette(index: usize) -> [f32; 3] {
    let mut value = index as u32 * 747796405 + 2891336453;
    value ^= value >> 16;
    value = value.wrapping_mul(2246822519);
    value ^= value >> 13;
    [
        0.25 + (value & 255) as f32 / 510.0,
        0.25 + ((value >> 8) & 255) as f32 / 510.0,
        0.25 + ((value >> 16) & 255) as f32 / 510.0,
    ]
}
