pub struct Button {
    pub rect: [f32; 4],
    pub label: &'static str,
    pub active: bool,
    pub pressed: bool,
}

pub struct Controls {
    pub mode: u32,
    pub scene: u32,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub auto_orbit: bool,
    pub trajectory: bool,
    pub playing: bool,
    pub mounted: bool,
    buttons: Vec<Button>,
    captured: bool,
    pressed: Option<usize>,
}

impl Controls {
    pub fn new() -> Self {
        Self {
            mode: 0,
            scene: 0,
            yaw: 0.65,
            pitch: 0.5,
            distance: 18.0,
            auto_orbit: false,
            trajectory: false,
            playing: false,
            mounted: false,
            buttons: Vec::with_capacity(10),
            captured: false,
            pressed: None,
        }
    }

    pub fn layout(&mut self, width: u32, _height: u32, pixel_ratio: f32) {
        let labels = [
            "RGB",
            "DEPTH",
            "IDS",
            "COMPARE",
            if self.trajectory {
                "SHOWCASE"
            } else if self.scene == 0 {
                "COURTYARD"
            } else {
                "CUBES"
            },
            "RESET",
            if self.trajectory && self.playing {
                "PAUSE"
            } else if self.trajectory {
                "PLAY"
            } else if self.auto_orbit {
                "ORBIT ON"
            } else {
                "ORBIT OFF"
            },
            "-",
            "+",
            if self.mounted { "MOUNTED" } else { "ORBIT" },
        ];
        let available = width as f32 / pixel_ratio;
        let mut x = 12.0;
        let mut y = 12.0;
        self.buttons.clear();
        for (index, label) in labels.into_iter().enumerate() {
            if index == 9 && !self.trajectory {
                break;
            }
            // Keep state-dependent labels at a stable size while a pointer is held.
            let characters = match index {
                4 | 6 | 9 => 9,
                _ => label.len(),
            };
            let button_width = (characters as f32 * 8.0 + 24.0).max(44.0);
            if index == 4 || (x > 12.0 && x + button_width > available - 12.0) {
                x = 12.0;
                y += 50.0;
            }
            self.buttons.push(Button {
                rect: [
                    x * pixel_ratio,
                    y * pixel_ratio,
                    button_width * pixel_ratio,
                    44.0 * pixel_ratio,
                ],
                label,
                active: index == self.mode as usize
                    || index == 4
                    || (index == 6
                        && if self.trajectory {
                            self.playing
                        } else {
                            self.auto_orbit
                        })
                    || (index == 9 && self.mounted),
                pressed: self.pressed == Some(index),
            });
            x += button_width + 6.0;
        }
    }

    pub fn buttons(&self) -> &[Button] {
        &self.buttons
    }

    fn hit(&self, x: f32, y: f32) -> Option<usize> {
        self.buttons.iter().position(|button| {
            let [left, top, width, height] = button.rect;
            x >= left && x < left + width && y >= top && y < top + height
        })
    }

    pub fn pointer_down(&mut self, x: f32, y: f32) -> bool {
        if self.captured {
            return true;
        }
        self.pressed = self.hit(x, y);
        self.captured = self.pressed.is_some();
        self.captured
    }

    pub fn pointer_move(&mut self, x: f32, y: f32) -> bool {
        if self.captured && self.hit(x, y) != self.pressed {
            self.pressed = None;
        }
        self.captured || self.hit(x, y).is_some()
    }

    pub fn pointer_up(&mut self, x: f32, y: f32) -> bool {
        let captured = self.captured;
        let pressed = self.pressed.take();
        self.captured = false;
        if let Some(index) = pressed.filter(|index| self.hit(x, y) == Some(*index)) {
            match index {
                0..=3 => self.set_mode(index as u32),
                4 => {
                    if self.trajectory {
                        self.trajectory = false;
                        self.playing = false;
                        self.reset();
                    } else {
                        self.set_scene(1 - self.scene);
                    }
                }
                5 => self.reset(),
                6 => {
                    if self.trajectory {
                        self.playing = !self.playing;
                    } else {
                        self.auto_orbit = !self.auto_orbit;
                    }
                }
                7 => self.zoom(1.1),
                8 => self.zoom(1.0 / 1.1),
                9 => self.mounted = !self.mounted,
                _ => unreachable!(),
            }
        }
        captured
    }

    pub fn pointer_cancel(&mut self) -> bool {
        let captured = self.captured;
        self.captured = false;
        self.pressed = None;
        captured
    }

    pub fn advance(&mut self, dt: f32) {
        if self.auto_orbit && dt.is_finite() {
            self.yaw += dt.clamp(0.0, 0.1) * 0.18;
        }
    }

    pub fn orbit_by(&mut self, dx: f32, dy: f32) {
        self.auto_orbit = false;
        self.yaw -= dx * 0.008;
        self.pitch = (self.pitch + dy * 0.006).clamp(0.08, 1.45);
    }

    pub fn zoom(&mut self, factor: f32) {
        self.distance = (self.distance * factor).clamp(4.0, 50.0);
    }

    pub fn reset(&mut self) {
        self.mounted = false;
        self.yaw = 0.65;
        self.pitch = 0.5;
        self.distance = 18.0;
        self.auto_orbit = false;
    }

    pub fn set_mode(&mut self, mode: u32) {
        self.mode = mode;
    }

    pub fn set_scene(&mut self, scene: u32) {
        self.trajectory = false;
        self.playing = false;
        if self.scene != scene {
            self.scene = scene;
            self.reset();
        }
    }

    pub fn key(&mut self, key: &str) -> bool {
        match key {
            "1" => self.set_mode(0),
            "2" => self.set_mode(1),
            "3" => self.set_mode(2),
            "4" => self.set_mode(3),
            "s" | "S" => self.set_scene(1 - self.scene),
            "r" | "R" => self.reset(),
            " " if self.trajectory => self.playing = !self.playing,
            " " => self.auto_orbit = !self.auto_orbit,
            "c" | "C" => self.mounted = !self.mounted,
            "ArrowLeft" => self.yaw -= 0.08,
            "ArrowRight" => self.yaw += 0.08,
            "ArrowUp" => self.pitch = (self.pitch + 0.06).clamp(0.08, 1.45),
            "ArrowDown" => self.pitch = (self.pitch - 0.06).clamp(0.08, 1.45),
            "+" | "=" => self.zoom(1.0 / 1.1),
            "-" | "_" => self.zoom(1.1),
            _ => return false,
        }
        true
    }
}
