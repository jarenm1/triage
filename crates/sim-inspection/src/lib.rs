use std::{
    collections::HashMap,
    fs,
    io::Write,
    path::Path,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail, ensure};
use font8x8::UnicodeFonts;
use glam::{Mat4, Quat, Vec3};
use image::{Rgba, RgbaImage};
use sim_graphics::{
    Camera, Frame, MeshData, MeshHandle, ReadbackData, RenderPrimitive, RenderView, Renderer,
    ViewKey, ViewKind, ViewOutputs,
};
use sim_scene::{ONTOLOGY_VERSION, SceneConfig, SceneObject, SensorConfig};

mod gpu;
pub use gpu::GpuInspector;

#[cfg(test)]
mod tests;

fn sensor_camera(s: &SensorConfig) -> Camera {
    Camera {
        eye: s.eye.into(),
        target: s.target.into(),
        up: s.up.into(),
        vertical_fov_radians: s.vertical_fov_radians,
        near: s.near,
        far: s.far,
    }
}

fn object_center(object: &SceneObject) -> Vec3 {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for part in &object.parts {
        let position = Vec3::from(part.position);
        let radius = Vec3::from(part.scale) * 0.5;
        min = min.min(position - radius);
        max = max.max(position + radius);
    }
    (min + max) * 0.5
}

pub struct Inspector {
    renderer: Renderer,
    cube: MeshHandle,
    frame: Frame,
    healthy: bool,
}

impl Inspector {
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let mut renderer = Renderer::new(&instance, None).await?;
        let cube = renderer.register_mesh(MeshData::cube())?;
        Ok(Self {
            renderer,
            cube,
            frame: Frame::with_capacity(64, 1),
            healthy: true,
        })
    }

    pub fn capture(&mut self, config: &SceneConfig) -> Result<Capture> {
        config.validate()?;
        ensure!(
            self.healthy,
            "inspector stopped after a failed GPU capture; create a new Inspector"
        );
        self.healthy = false;
        let result = self.capture_inner(config);
        self.healthy = result.is_ok();
        result
    }

    fn scene(&mut self, config: &SceneConfig) {
        self.frame.begin();
        for o in &config.objects {
            for part in &o.parts {
                self.frame.draw(RenderPrimitive {
                    mesh: self.cube,
                    transform: Mat4::from_scale_rotation_translation(
                        part.scale.into(),
                        Quat::IDENTITY,
                        part.position.into(),
                    ),
                    color: part.color,
                    object_id: o.id,
                    pattern: [0.0; 4],
                    aux: [0.0; 4],
                });
            }
        }
    }

    fn render(
        &mut self,
        camera: Camera,
        width: u32,
        height: u32,
        all_outputs: bool,
    ) -> Result<Vec<ReadbackData>> {
        let outputs = if all_outputs {
            ViewOutputs::COLOR | ViewOutputs::DEPTH | ViewOutputs::OBJECT_ID
        } else {
            ViewOutputs::COLOR
        };
        self.frame.add_view(RenderView {
            key: ViewKey(if all_outputs { 1 } else { 2 }),
            kind: ViewKind::Sensor,
            camera,
            width,
            height,
            outputs,
        });
        let submission = self.renderer.execute(&self.frame, &[])?;
        ensure!(
            submission.readbacks.len() == if all_outputs { 3 } else { 1 },
            "renderer omitted requested inspection readbacks"
        );
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut results = Vec::with_capacity(submission.readbacks.len());
        for handle in submission.readbacks {
            loop {
                if let Some(data) = self.renderer.poll_readback(handle)? {
                    results.push(data);
                    break;
                }
                ensure!(
                    Instant::now() < deadline,
                    "GPU inspection readback timed out after 30 seconds"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        Ok(results)
    }

    fn line(&mut self, a: Vec3, b: Vec3, thickness: f32, color: [f32; 4]) {
        let delta = b - a;
        if delta.length_squared() < 1e-12 {
            return;
        }
        self.frame.draw(RenderPrimitive {
            mesh: self.cube,
            transform: Mat4::from_scale_rotation_translation(
                Vec3::new(thickness, thickness, delta.length()),
                Quat::from_rotation_arc(Vec3::Z, delta.normalize()),
                (a + b) * 0.5,
            ),
            color,
            object_id: 0,
                    pattern: [0.0; 4],
                    aux: [0.0; 4],
        });
    }

    fn observer_scene(&mut self, config: &SceneConfig) -> Camera {
        let s = &config.sensor;
        self.scene(config);
        let corners = frustum_corners(s);
        let eye = Vec3::from(s.eye);
        let mut min = eye.min(Vec3::ZERO);
        let mut max = eye.max(Vec3::splat(2.0));
        for p in corners {
            min = min.min(p);
            max = max.max(p);
        }
        for o in &config.objects {
            for part in &o.parts {
                let p = Vec3::from(part.position);
                let r = Vec3::from(part.scale) * 0.5;
                min = min.min(p - r);
                max = max.max(p + r);
            }
        }
        let center = (min + max) * 0.5;
        let radius = ((max - min).length() * 0.5).max(1.0);
        let thickness = radius * 0.007;
        for i in 0..4 {
            self.line(
                corners[i],
                corners[(i + 1) % 4],
                thickness,
                [1.0, 0.65, 0.04, 1.0],
            );
            self.line(
                corners[i + 4],
                corners[(i + 1) % 4 + 4],
                thickness,
                [1.0, 0.65, 0.04, 1.0],
            );
            self.line(eye, corners[i + 4], thickness, [1.0, 0.65, 0.04, 1.0]);
        }
        for (axis, c) in [
            (Vec3::X, [1.0, 0.05, 0.05, 1.0]),
            (Vec3::Y, [0.05, 1.0, 0.05, 1.0]),
            (Vec3::Z, [0.1, 0.3, 1.0, 1.0]),
        ] {
            self.line(Vec3::ZERO, axis * 2.0, thickness * 1.5, c);
        }
        let observer = Camera {
            eye: center + Vec3::new(1.2, 0.85, 1.5).normalize() * radius * 2.5,
            target: center,
            up: Vec3::Y,
            vertical_fov_radians: 50_f32.to_radians(),
            near: (radius * 0.001).max(0.001),
            far: radius * 8.0,
        };
        observer
    }

    fn capture_inner(&mut self, config: &SceneConfig) -> Result<Capture> {
        self.scene(config);
        let s = &config.sensor;
        let mut color = None;
        let mut depth = None;
        let mut object_ids = None;
        for data in self.render(sensor_camera(s), s.width, s.height, true)? {
            match data {
                ReadbackData::Color(v) => color = Some(v),
                ReadbackData::Depth(v) => depth = Some(v),
                ReadbackData::ObjectIds(v) => object_ids = Some(v),
            }
        }
        let color = color.context("missing RGB output")?;
        let depth = depth.context("missing depth output")?;
        let object_ids = object_ids.context("missing ID output")?;
        let count = s.width as usize * s.height as usize;
        ensure!(
            color.len() == count * 4 && depth.len() == count && object_ids.len() == count,
            "renderer returned incorrect capture dimensions"
        );

        // The second submission is an independent observer scene. All sensor readbacks
        // above are consumed before its gizmos exist, so they cannot enter sensor truth.
        let observer = self.observer_scene(config);
        let eye = Vec3::from(s.eye);
        let observer_width = 640;
        let observer_height = 480;
        let observer_data = self
            .render(observer, observer_width, observer_height, false)?
            .pop()
            .context("missing observer output")?;
        let ReadbackData::Color(observer_bytes) = observer_data else {
            bail!("unexpected observer output kind")
        };
        let mut overview = display_color(&observer_bytes, observer_width, observer_height)?;
        for (index, o) in config.objects.iter().enumerate() {
            label_world(
                &mut overview,
                observer,
                object_center(o),
                &format!("{} {}", o.id, o.name),
                id_color(o.id),
                [8, (16 + index as u32 * 18).min(420)],
            );
        }
        label_world(
            &mut overview,
            observer,
            eye,
            "SENSOR",
            [255, 220, 70, 255],
            [8, 452],
        );
        for (index, (p, label, c)) in [
            (Vec3::X * 2.0, "+X", [255, 80, 80, 255]),
            (Vec3::Y * 2.0, "+Y", [80, 255, 80, 255]),
            (Vec3::Z * 2.0, "+Z", [90, 130, 255, 255]),
        ]
        .into_iter()
        .enumerate()
        {
            label_world(
                &mut overview,
                observer,
                p,
                label,
                c,
                [570, 16 + index as u32 * 20],
            );
        }
        let preview = compose(config, &color, &depth, &object_ids, overview)?;
        Ok(Capture {
            preview,
            color,
            depth,
            object_ids,
            config: config.clone(),
        })
    }
}

fn frustum_corners(s: &SensorConfig) -> [Vec3; 8] {
    let c = sensor_camera(s);
    let forward = (c.target - c.eye).normalize();
    let right = forward.cross(c.up).normalize();
    let up = right.cross(forward);
    let mut corners = [Vec3::ZERO; 8];
    for (plane, d) in [s.near, s.far].into_iter().enumerate() {
        let h = d * (s.vertical_fov_radians * 0.5).tan();
        let w = h * s.width as f32 / s.height as f32;
        for (i, (x, y)) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
            .into_iter()
            .enumerate()
        {
            corners[plane * 4 + i] = c.eye + forward * d + right * w * x + up * h * y;
        }
    }
    corners
}

pub struct Capture {
    pub preview: RgbaImage,
    /// Top-to-bottom RGBA8 UNORM, linear RGB lighting clipped to [0,1], NOT sRGB.
    pub color: Vec<u8>,
    /// Optical +Z depth in metres; background equals the sensor's far plane.
    pub depth: Vec<f32>,
    /// Full stable uint32 instance identifiers; zero marks background.
    pub object_ids: Vec<u32>,
    config: SceneConfig,
}

impl Capture {
    pub fn save(&self, config: &SceneConfig, directory: impl AsRef<Path>) -> Result<()> {
        config.validate()?;
        ensure!(
            *config == self.config,
            "capture/config mismatch: save with the exact configuration used to capture"
        );
        let count = config.sensor.width as usize * config.sensor.height as usize;
        ensure!(
            self.color.len() == count * 4
                && self.depth.len() == count
                && self.object_ids.len() == count,
            "capture payload dimensions no longer match configuration"
        );
        let classes: HashMap<u32, u32> = config
            .objects
            .iter()
            .map(|object| (object.id, object.class.id()))
            .chain([(0, 0)])
            .collect();
        ensure!(
            self.object_ids.iter().all(|id| classes.contains_key(id)),
            "capture instance mask contains an ID absent from the realized scene"
        );
        let directory = directory.as_ref();
        // Publish the whole bundle with one rename: never overwrite an earlier capture
        // with a half-written mixture of frames. Existing empty output dirs are accepted.
        let parent = directory
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::create_dir_all(parent)?;
        if directory.exists() {
            ensure!(
                directory.is_dir() && fs::read_dir(directory)?.next().is_none(),
                "capture destination {} must be absent or empty",
                directory.display()
            );
        }
        let staging = tempfile::tempdir_in(parent)?;
        self.preview.save(staging.path().join("preview.png"))?;
        fs::write(staging.path().join("color.rgba8"), &self.color)?;
        display_color(&self.color, config.sensor.width, config.sensor.height)?
            .save(staging.path().join("color.png"))?;
        write_words(
            &staging.path().join("semantic_classes.u32le"),
            self.object_ids.iter().map(|id| classes[id].to_le_bytes()),
        )?;
        write_words(
            &staging.path().join("depth.f32le"),
            self.depth.iter().map(|v| v.to_le_bytes()),
        )?;
        write_words(
            &staging.path().join("object_ids.u32le"),
            self.object_ids.iter().map(|v| v.to_le_bytes()),
        )?;
        config.save(staging.path().join("scene.json"))?;
        let metadata = serde_json::json!({
            "version": 2, "scene": config,
            "width": config.sensor.width, "height": config.sensor.height,
            "layout": "row-major, top row first, no row padding", "world": "right-handed, Y up; metres",
            "intrinsics": config.sensor.intrinsics(), "pixel_coordinates": "origin upper-left edge; pixel centers (column+0.5,row+0.5)",
            "world_to_optical_column_major": config.sensor.world_to_optical(),
            "optical_axes": "X right, Y down, Z forward", "distortion": "none",
            "color": {"file":"color.rgba8", "format":"RGBA8 UNORM", "transfer_function":"linear (not sRGB); shader lit_color clamped and quantized to 8-bit UNORM", "channels":4},
            "color_srgb": {"file":"color.png", "format":"RGBA8 PNG", "transfer_function":"sRGB", "source":"color.rgba8: the same clipped, quantized linear LDR sensor output; not HDR"},
            "depth": {"file":"depth.f32le", "format":"IEEE754 float32 little-endian", "quantity":"optical +Z, not Euclidean range", "units":"metres", "background":config.sensor.far},
            "object_ids": {"file":"object_ids.u32le", "format":"uint32 little-endian", "background":0},
            "semantic_classes": {"file":"semantic_classes.u32le", "format":"uint32 little-endian", "background":0},
            "semantic_ontology": {"version": ONTOLOGY_VERSION, "classes": [
                {"id":0, "name":"background"}, {"id":1, "name":"ground"},
                {"id":2, "name":"gate"}, {"id":3, "name":"obstacle"}
            ]},
            "instances": config.objects.iter().map(|object| serde_json::json!({
                "id":object.id, "name":object.name, "class":object.class, "class_id":object.class.id()
            })).collect::<Vec<_>>(),
            "preview": "preview.png is diagnostic only: linear RGB encoded to sRGB for display; depth normalized near..far; ID palette is not truth; observer gizmos excluded from all sensor payloads"
        });
        fs::write(
            staging.path().join("metadata.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )?;
        fs::rename(staging.path(), directory)
            .with_context(|| format!("publishing capture {}", directory.display()))?;
        Ok(())
    }
}

fn write_words(path: &Path, values: impl Iterator<Item = [u8; 4]>) -> Result<()> {
    let mut writer = std::io::BufWriter::new(fs::File::create(path)?);
    for value in values {
        writer.write_all(&value)?;
    }
    writer.flush()?;
    Ok(())
}

fn display_color(bytes: &[u8], width: u32, height: u32) -> Result<RgbaImage> {
    let mut image = RgbaImage::from_raw(width, height, bytes.to_vec())
        .context("invalid color buffer dimensions")?;
    for p in image.pixels_mut() {
        for c in &mut p.0[..3] {
            let linear = *c as f32 / 255.0;
            *c = (255.0
                * if linear <= 0.0031308 {
                    12.92 * linear
                } else {
                    1.055 * linear.powf(1.0 / 2.4) - 0.055
                })
            .round() as u8;
        }
        p.0[3] = 255;
    }
    Ok(image)
}

fn id_color(id: u32) -> [u8; 4] {
    if id == 0 {
        return [0, 0, 0, 255];
    }
    let mut hash = id;
    hash = (hash ^ (hash >> 16)).wrapping_mul(0x7feb352d);
    hash = (hash ^ (hash >> 15)).wrapping_mul(0x846ca68b);
    hash ^= hash >> 16;
    [
        64 + ((hash & 255) * 191 / 255) as u8,
        64 + (((hash >> 8) & 255) * 191 / 255) as u8,
        64 + (((hash >> 16) & 255) * 191 / 255) as u8,
        255,
    ]
}

fn text(image: &mut RgbaImage, x: u32, y: u32, value: &str, color: [u8; 4]) {
    for (i, ch) in value.chars().enumerate() {
        let x = x + i as u32 * 16;
        if x + 16 > image.width() {
            break;
        }
        if let Some(glyph) = font8x8::BASIC_FONTS
            .get(ch)
            .or_else(|| font8x8::BASIC_FONTS.get('?'))
        {
            for dy in 0..16 {
                for dx in 0..16 {
                    if y + dy < image.height() {
                        let filled = glyph[(dy / 2) as usize] & (1 << (dx / 2)) != 0;
                        image.put_pixel(
                            x + dx,
                            y + dy,
                            Rgba(if filled { color } else { [16, 19, 25, 255] }),
                        );
                    }
                }
            }
        }
    }
}

fn label_world(
    image: &mut RgbaImage,
    camera: Camera,
    point: Vec3,
    label: &str,
    color: [u8; 4],
    anchor: [u32; 2],
) {
    let clip =
        camera.view_projection(image.width() as f32 / image.height() as f32) * point.extend(1.0);
    if clip.w <= 0.0 {
        return;
    }
    let ndc = clip.truncate() / clip.w;
    if ndc.x.abs() > 1.0 || ndc.y.abs() > 1.0 || !(0.0..=1.0).contains(&ndc.z) {
        return;
    }
    let x = ((ndc.x + 1.0) * 0.5 * image.width() as f32) as u32;
    let y = ((1.0 - ndc.y) * 0.5 * image.height() as f32) as u32;
    let start_x = (anchor[0] + label.chars().count() as u32 * 16).min(image.width() - 1);
    let start_y = (anchor[1] + 6).min(image.height() - 1);
    let end_x = x.min(image.width() - 1);
    let end_y = y.min(image.height() - 1);
    let steps = start_x.abs_diff(end_x).max(start_y.abs_diff(end_y)).max(1);
    for step in 0..=steps {
        let t = step as f32 / steps as f32;
        let px = (start_x as f32 * (1.0 - t) + end_x as f32 * t).round() as u32;
        let py = (start_y as f32 * (1.0 - t) + end_y as f32 * t).round() as u32;
        image.put_pixel(
            px.min(image.width() - 1),
            py.min(image.height() - 1),
            Rgba(color),
        );
    }
    text(image, anchor[0], anchor[1], label, color);
}

fn compose(
    config: &SceneConfig,
    color: &[u8],
    depth: &[f32],
    ids: &[u32],
    overview: RgbaImage,
) -> Result<RgbaImage> {
    let s = &config.sensor;
    let rgb = display_color(color, s.width, s.height)?;
    let mut depth_image = RgbaImage::new(s.width, s.height);
    let mut id_image = RgbaImage::new(s.width, s.height);
    for (index, p) in depth_image.pixels_mut().enumerate() {
        let d = depth[index];
        *p = Rgba(if ids[index] == 0 {
            [0, 0, 0, 255]
        } else if !d.is_finite() {
            [255, 0, 255, 255]
        } else {
            let t = ((d - s.near) / (s.far - s.near)).clamp(0.0, 1.0);
            [
                (255.0 * (1.0 - t)) as u8,
                (255.0 * (1.0 - (2.0 * t - 1.0).abs())) as u8,
                (255.0 * t) as u8,
                255,
            ]
        });
    }
    for (p, id) in id_image.pixels_mut().zip(ids) {
        *p = Rgba(id_color(*id));
    }
    let footer = 128 + config.objects.len() as u32 * 20;
    let mut preview = RgbaImage::from_pixel(1280, 1024 + footer, Rgba([16, 19, 25, 255]));
    for (index, (title, panel)) in [
        ("RGB | sRGB preview (linear truth)", rgb),
        ("DEPTH | optical Z: near -> far", depth_image),
        ("INSTANCE IDs | uint32 truth", id_image),
        ("OVERVIEW | frustum + world axes", overview),
    ]
    .into_iter()
    .enumerate()
    {
        let x = (index as u32 % 2) * 640;
        let y = (index as u32 / 2) * 512;
        text(&mut preview, x + 8, y + 10, title, [235, 240, 250, 255]);
        let scale = (640.0 / panel.width() as f32).min(480.0 / panel.height() as f32);
        let w = (panel.width() as f32 * scale).round().max(1.0) as u32;
        let h = (panel.height() as f32 * scale).round().max(1.0) as u32;
        let panel = image::imageops::resize(&panel, w, h, image::imageops::FilterType::Nearest);
        image::imageops::replace(
            &mut preview,
            &panel,
            (x + (640 - w) / 2) as i64,
            (y + 32 + (480 - h) / 2) as i64,
        );
    }
    let k = s.intrinsics();
    text(
        &mut preview,
        8,
        1032,
        &format!(
            "{}x{} | fx=fy {:.2} | cx={:.2} cy={:.2} | FOVy {:.2}deg",
            s.width,
            s.height,
            k[0][0],
            k[0][2],
            k[1][2],
            s.vertical_fov_radians.to_degrees()
        ),
        [240, 240, 240, 255],
    );
    text(
        &mut preview,
        8,
        1052,
        &format!(
            "Near {:.3}m Far {:.3}m | RH world: Y up (metres)",
            s.near, s.far
        ),
        [210, 215, 225, 255],
    );
    text(
        &mut preview,
        8,
        1072,
        "Optical: X right / Y down / Z forward | background ID=0 depth=far",
        [210, 215, 225, 255],
    );
    text(
        &mut preview,
        8,
        1092,
        "RGB truth: linear UNORM; preview: sRGB. Frustum: true near/far.",
        [210, 215, 225, 255],
    );
    text(
        &mut preview,
        8,
        1112,
        "Gizmos are observer-only. Palette colors are not instance-ID data.",
        [210, 215, 225, 255],
    );
    for (i, o) in config.objects.iter().enumerate() {
        text(
            &mut preview,
            8,
            1136 + i as u32 * 20,
            &format!("ID {}: {}", o.id, o.name),
            id_color(o.id),
        );
    }
    Ok(preview)
}
