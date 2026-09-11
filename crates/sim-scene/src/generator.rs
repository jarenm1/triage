use super::*;

pub const GENERATOR_VERSION: u32 = 1;

/// Versioned, inclusive uniform integer distributions. Geometry is realized in metres;
/// integer millimetres and an explicit wrapping RNG make sampling target-independent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GeneratorRecipe {
    pub version: u32,
    pub seed: u32,
    pub width: u32,
    pub height: u32,
    pub obstacle_count: [u32; 2],
    pub obstacle_size_mm: [u32; 2],
    pub obstacle_spread_mm: u32,
    pub gate_width_mm: [u32; 2],
    pub gate_height_mm: [u32; 2],
    pub camera_jitter_mm: u32,
}

impl Default for GeneratorRecipe {
    fn default() -> Self {
        Self {
            version: GENERATOR_VERSION,
            seed: 1,
            width: 640,
            height: 480,
            obstacle_count: [4, 9],
            obstacle_size_mm: [400, 1400],
            obstacle_spread_mm: 3500,
            gate_width_mm: [2200, 3400],
            gate_height_mm: [2400, 3600],
            camera_jitter_mm: 800,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Generation {
    pub recipe: GeneratorRecipe,
    pub sample_index: u32,
}

impl GeneratorRecipe {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == GENERATOR_VERSION,
            "unsupported generator version {}",
            self.version
        );
        ensure!(
            (1..=2048).contains(&self.width) && (1..=2048).contains(&self.height),
            "generator dimensions must be 1..=2048"
        );
        for (name, range, min, max) in [
            ("obstacle_count", self.obstacle_count, 0, 24),
            ("obstacle_size_mm", self.obstacle_size_mm, 200, 1800),
            ("gate_width_mm", self.gate_width_mm, 1600, 4000),
            ("gate_height_mm", self.gate_height_mm, 1800, 4200),
        ] {
            ensure!(
                min <= range[0] && range[0] <= range[1] && range[1] <= max,
                "{name} requires {min} <= lower <= upper <= {max}"
            );
        }
        ensure!(
            (2500..=6000).contains(&self.obstacle_spread_mm),
            "obstacle_spread_mm must be 2500..=6000"
        );
        ensure!(
            self.camera_jitter_mm <= 2000,
            "camera_jitter_mm must be <= 2000"
        );
        Ok(())
    }

    /// Stateless by sample index: batching, seeking, and generation order cannot affect output.
    pub fn generate(&self, sample_index: u32) -> Result<SceneConfig> {
        self.validate()?;
        let mut rng = Rng(mix(self.seed) ^ mix(sample_index.wrapping_add(0x9e3779b9)));
        let mut objects = Vec::with_capacity(self.obstacle_count[1] as usize + 3);
        objects.push(SceneObject {
            id: 1,
            name: "ground".into(),
            class: SemanticClass::Ground,
            parts: vec![part(
                [0, -100, -6000],
                [18000, 200, 32000],
                [0.22, 0.27, 0.20, 1.0],
            )],
        });
        for (index, (x, z)) in [(-2500, -1500), (2500, -6500)].into_iter().enumerate() {
            let width = rng.range(self.gate_width_mm) as i32;
            let height = rng.range(self.gate_height_mm) as i32;
            let color = if index == 0 {
                [0.85, 0.35, 0.05, 1.0]
            } else {
                [0.08, 0.4, 0.85, 1.0]
            };
            objects.push(SceneObject {
                id: 0x8000_0001 + index as u32,
                name: format!("gate {}", index + 1),
                class: SemanticClass::Gate,
                parts: vec![
                    part([x - width / 2, height / 2, z], [200, height, 300], color),
                    part([x + width / 2, height / 2, z], [200, height, 300], color),
                    part([x, height + 100, z], [width + 200, 200, 300], color),
                ],
            });
        }
        // Distinct shuffled cells keep boxes separated even at maximum size. They
        // may occlude gates, intentionally; no rejection loop or sample history exists.
        let mut cells: Vec<u32> = (0..24).collect();
        let count = rng.range(self.obstacle_count);
        for index in 0..count {
            let chosen = rng.range([index, 23]) as usize;
            cells.swap(index as usize, chosen);
            let cell = cells[index as usize];
            let x = (cell % 3) as i32 - 1;
            let z = (cell / 3) as i32;
            let size = [
                rng.range(self.obstacle_size_mm) as i32,
                rng.range(self.obstacle_size_mm) as i32,
                rng.range(self.obstacle_size_mm) as i32,
            ];
            let color = [
                rng.range([80, 850]) as f32 / 1000.0,
                rng.range([80, 850]) as f32 / 1000.0,
                rng.range([80, 850]) as f32 / 1000.0,
                1.0,
            ];
            objects.push(SceneObject {
                id: 0xffff_0000 + index,
                name: format!("obstacle {}", index + 1),
                class: SemanticClass::Obstacle,
                parts: vec![part(
                    [
                        x * self.obstacle_spread_mm as i32 + rng.signed(200),
                        size[1] / 2,
                        4500 - z * 3000 + rng.signed(200),
                    ],
                    size,
                    color,
                )],
            });
        }
        let mut sensor = SensorConfig {
            eye: [
                rng.signed(self.camera_jitter_mm) as f32 / 1000.0,
                5.0 + rng.signed(self.camera_jitter_mm) as f32 / 1000.0,
                14.0 + rng.signed(self.camera_jitter_mm) as f32 / 1000.0,
            ],
            target: [
                rng.signed(self.camera_jitter_mm / 2) as f32 / 1000.0,
                1.0,
                -4.0,
            ],
            up: [0.0, 1.0, 0.0],
            vertical_fov_radians: 1.0,
            near: 0.1,
            far: 60.0,
            width: self.width,
            height: self.height,
        };
        // Sample radians directly on a fixed grid, without platform transcendental math.
        sensor.vertical_fov_radians = rng.range([850, 1100]) as f32 / 1000.0;
        let scene = SceneConfig {
            version: SCENE_VERSION,
            sensor,
            objects,
            generation: Some(Generation {
                recipe: self.clone(),
                sample_index,
            }),
        };
        scene.validate()?;
        Ok(scene)
    }
}

fn part(position: [i32; 3], scale: [i32; 3], color: [f32; 4]) -> BoxPart {
    BoxPart {
        position: position.map(|v| v as f32 / 1000.0),
        scale: scale.map(|v| v as f32 / 1000.0),
        color,
    }
}
fn mix(mut v: u32) -> u32 {
    v = (v ^ (v >> 16)).wrapping_mul(0x7feb352d);
    v = (v ^ (v >> 15)).wrapping_mul(0x846ca68b);
    v ^ (v >> 16)
}
struct Rng(u32);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9e3779b9);
        mix(self.0)
    }
    fn range(&mut self, [min, max]: [u32; 2]) -> u32 {
        let span = max - min + 1;
        let threshold = span.wrapping_neg() % span;
        loop {
            let value = self.next();
            if value >= threshold {
                return min + value % span;
            }
        }
    }
    fn signed(&mut self, radius: u32) -> i32 {
        self.range([0, radius * 2]) as i32 - radius as i32
    }
}
