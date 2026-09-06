use anyhow::{Context, Result, ensure};
use parking_lot::Mutex;
use sim_graphics::{Frame, RenderView, Renderer, RendererError, ViewKey, ViewKind, ViewOutputs};
use sim_graphics_winit::{
    Scene,
    winit::{
        event::{ElementState, MouseScrollDelta, WindowEvent},
        keyboard::{KeyCode, PhysicalKey},
    },
};
use sim_trajectory::{CameraMode, Header, Recording, RenderSnapshot, TrajectoryScene};
use std::{
    io::{BufRead, BufReader, Read},
    net::TcpStream,
    path::PathBuf,
    sync::Arc,
    thread,
    time::Instant,
};

const MAX_LINE_BYTES: u64 = 128 * 1024;
#[derive(Default)]
struct Mailbox {
    header: Option<Header>,
    snapshot: Option<RenderSnapshot>,
    status: Option<String>,
}

enum Source {
    Replay(Recording),
    Live(Arc<Mutex<Mailbox>>),
}

fn line(reader: &mut impl BufRead) -> Result<Option<String>> {
    let mut text = String::new();
    let bytes = reader.take(MAX_LINE_BYTES + 1).read_line(&mut text)?;
    if bytes == 0 {
        return Ok(None);
    }
    ensure!(
        bytes as u64 <= MAX_LINE_BYTES && text.ends_with('\n'),
        "incomplete or oversized live message"
    );
    Ok(Some(text))
}

pub fn run(live: bool, value: std::ffi::OsString) -> Result<()> {
    let source = if live {
        let address = value
            .into_string()
            .map_err(|_| anyhow::anyhow!("live address must be UTF-8"))?;
        let mailbox = Arc::new(Mutex::new(Mailbox::default()));
        let output = mailbox.clone();
        thread::spawn(move || {
            let result = (|| -> Result<()> {
                let mut reader = BufReader::new(
                    TcpStream::connect(&address).context("connecting trajectory producer")?,
                );
                let header = Header::parse(&line(&mut reader)?.context("missing live header")?)?;
                output.lock().header = Some(header.clone());
                let mut previous = None;
                while let Some(text) = line(&mut reader)? {
                    let snapshot = RenderSnapshot::parse(&text, &header, previous.as_ref())?;
                    output.lock().snapshot = Some(snapshot.clone());
                    previous = Some(snapshot);
                }
                Ok(())
            })();
            output.lock().status = Some(match result {
                Ok(()) => "live stream ended".into(),
                Err(error) => format!("live stream error: {error:#}"),
            });
        });
        Source::Live(mailbox)
    } else {
        let path = PathBuf::from(value);
        Source::Replay(Recording::parse(
            &std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?,
        )?)
    };
    eprintln!(
        "Trajectory controls: Space pause, R restart, C orbit/mounted, N next vehicle, arrows orbit, wheel zoom. Telemetry uses simulation timestamps; no pose interpolation."
    );
    let (header, snapshot) = match &source {
        Source::Replay(recording) => (
            Some(recording.header.clone()),
            Some(recording.snapshots[0].clone()),
        ),
        Source::Live(_) => (None, None),
    };
    let viewer = Viewer {
        source,
        header,
        snapshot,
        scene: TrajectoryScene::new(),
        frame: Frame::with_capacity(1024, 1),
        clock: Instant::now(),
        play_seconds: 0.0,
        paused: false,
        mounted: false,
        selected: 0,
        azimuth: 0.8,
        elevation: 0.55,
        distance: 9.0,
        telemetry: Instant::now(),
    };
    Ok(sim_graphics_winit::run(
        "Trained policy trajectory — Space pause | R restart | C camera | N vehicle",
        viewer,
    )?)
}

struct Viewer {
    source: Source,
    header: Option<Header>,
    snapshot: Option<RenderSnapshot>,
    scene: TrajectoryScene,
    frame: Frame,
    clock: Instant,
    play_seconds: f64,
    paused: bool,
    mounted: bool,
    selected: usize,
    azimuth: f32,
    elevation: f32,
    distance: f32,
    telemetry: Instant,
}

impl Scene for Viewer {
    fn initialize(&mut self, renderer: &mut Renderer) -> Result<(), RendererError> {
        self.scene.initialize(renderer)
    }

    fn frame(&mut self, _elapsed_seconds: f32, width: u32, height: u32) -> &Frame {
        let preparation = Instant::now();
        let delta = self.clock.elapsed().as_secs_f64();
        self.clock = Instant::now();
        if !self.paused {
            self.play_seconds += delta;
        }
        match &self.source {
            Source::Replay(recording) => {
                self.play_seconds = self.play_seconds.min(recording.duration());
                let current =
                    recording.sample(recording.snapshots[0].time_seconds + self.play_seconds);
                if self
                    .snapshot
                    .as_ref()
                    .is_none_or(|old| old.step != current.step)
                {
                    match &mut self.snapshot {
                        Some(snapshot) => snapshot.clone_from(current),
                        None => self.snapshot = Some(current.clone()),
                    }
                }
            }
            Source::Live(mailbox) => {
                // Never wait for the network worker. Only one complete latest snapshot is retained.
                if let Some(mut mailbox) = mailbox.try_lock() {
                    if self.header.is_none() {
                        self.header = mailbox.header.take();
                    }
                    if !self.paused {
                        if let Some(snapshot) = mailbox.snapshot.take() {
                            self.snapshot = Some(snapshot);
                        }
                    }
                    if let Some(status) = mailbox.status.take() {
                        eprintln!("{status}");
                    }
                }
            }
        }
        self.frame.begin();
        if let (Some(header), Some(snapshot)) = (&self.header, &self.snapshot) {
            let selected = header.environment_ids[self.selected % header.environment_ids.len()];
            let mode = if self.mounted {
                CameraMode::Mounted {
                    environment_id: selected,
                }
            } else {
                CameraMode::Orbit {
                    azimuth: self.azimuth,
                    elevation: self.elevation,
                    distance: self.distance,
                }
            };
            self.frame.add_view(RenderView {
                key: ViewKey(0),
                kind: ViewKind::Display,
                camera: self.scene.camera(header, snapshot, mode),
                width,
                height,
                outputs: ViewOutputs::COLOR,
            });
            self.scene.draw(&mut self.frame, snapshot);
            if self.telemetry.elapsed().as_secs_f32() >= 1.0 {
                let vehicle = snapshot
                    .vehicles
                    .iter()
                    .find(|v| v.environment_id == selected)
                    .unwrap();
                eprintln!(
                    "trajectory step={} t={:.3}s env={} episode={} position_ENU={:?} paused={} CPU_prepare_us={}",
                    snapshot.step,
                    snapshot.time_seconds,
                    selected,
                    vehicle.episode_id,
                    vehicle.position_w,
                    self.paused,
                    preparation.elapsed().as_micros()
                );
                self.telemetry = Instant::now();
            }
        }
        if self.frame.views().is_empty() {
            self.frame.add_view(RenderView {
                key: ViewKey(0),
                kind: ViewKind::Display,
                camera: sim_graphics::Camera {
                    eye: glam::Vec3::new(5.0, 5.0, 5.0),
                    target: glam::Vec3::ZERO,
                    up: glam::Vec3::Y,
                    vertical_fov_radians: 0.87266463,
                    near: 0.1,
                    far: 80.0,
                },
                width,
                height,
                outputs: ViewOutputs::COLOR,
            });
        }
        &self.frame
    }

    fn window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed && !event.repeat =>
            {
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Space) => self.paused = !self.paused,
                    PhysicalKey::Code(KeyCode::KeyR) => self.play_seconds = 0.0,
                    PhysicalKey::Code(KeyCode::KeyC) => self.mounted = !self.mounted,
                    PhysicalKey::Code(KeyCode::KeyN) => {
                        self.selected = self.selected.wrapping_add(1)
                    }
                    PhysicalKey::Code(KeyCode::ArrowLeft) => self.azimuth -= 0.15,
                    PhysicalKey::Code(KeyCode::ArrowRight) => self.azimuth += 0.15,
                    PhysicalKey::Code(KeyCode::ArrowUp) => {
                        self.elevation = (self.elevation + 0.1).min(1.5)
                    }
                    PhysicalKey::Code(KeyCode::ArrowDown) => {
                        self.elevation = (self.elevation - 0.1).max(0.05)
                    }
                    _ => {}
                }
                self.clock = Instant::now();
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let delta = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.02,
                };
                self.distance = (self.distance * (-delta * 0.1).exp()).clamp(1.0, 50.0);
            }
            _ => {}
        }
    }
}
