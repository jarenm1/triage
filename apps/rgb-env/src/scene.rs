//! Procedural scene generation: pure functions of a seed producing dense,
//! packed geometry. Every generator is f(seed, index) -> primitives.
//!
//! Layout: a corridor the vehicle flies through, surrounded by dense
//! procedural scenery — buildings, branching trees, wire/fence structures,
//! ground cover — all derived from the seed.

use glam::{Quat, Vec3};

/// A generated primitive: position, scale, rotation, color, pattern, mesh.
#[derive(Clone, Copy)]
pub struct Prim {
    pub pos: Vec3,
    pub scale: Vec3,
    pub rot: Quat,
    pub color: [f32; 4],
    pub pattern: [f32; 4],
    pub mesh: MeshKind,
    /// Aux: (snap_flag, base_x, base_z, unused).
    pub aux: [f32; 4],
}

#[derive(Clone, Copy, PartialEq)]
pub enum MeshKind {
    Cube,
    Cylinder,
    Sphere,
    Terrain,
}

fn mix64(mut v: u64) -> u64 {
    v ^= v >> 30;
    v = v.wrapping_mul(0xbf58476d1ce4e5b9);
    v ^= v >> 27;
    v = v.wrapping_mul(0x94d049bb133111eb);
    v ^ v >> 31
}

fn rng(seed: u64, index: u64) -> f32 {
    (mix64(seed.wrapping_add(index.wrapping_mul(0xc2b2ae35))) & 1023) as f32
        / 1023.0
}

fn color(seed: u64, index: u64) -> [f32; 4] {
    [
        0.08 + rng(seed, index) * 0.80,
        0.08 + rng(seed, index + 100) * 0.80,
        0.08 + rng(seed, index + 200) * 0.80,
        1.0,
    ]
}

fn pattern(seed: u64, index: u64) -> [f32; 4] {
    // always textured — heavy procedural patterns for domain randomization
    let kind = (rng(seed, index + 301) * 8.0).floor() + 1.0;
    [
        kind,
        0.5 + rng(seed, index + 302) * 4.0,
        rng(seed, index + 303) * 100.0,
        0.3 + rng(seed, index + 304) * 0.6,
    ]
}

/// The task wall: two boxes with a gap. Returns (prims, gap_x).
pub fn wall(seed: u64) -> ([Prim; 2], f32) {
    let gap_x = (rng(seed, 4) - 0.5) * 5.2;
    let gap_half = 1.5f32;
    let wall_z = -5.2f32;
    let wall_half_z = 0.15f32;
    let wall_y = 1.2f32;
    let corridor_half = 4.5f32;
    let left_w = (gap_x - gap_half) - (-corridor_half);
    let right_w = corridor_half - (gap_x + gap_half);
    let mk = |pos: Vec3, scale: Vec3, i: u64| Prim {
        pos,
        scale,
        rot: Quat::IDENTITY,
        color: color(seed, i),
        pattern: pattern(seed, i),
        mesh: MeshKind::Cube,
        aux: [0.0; 4],
    };
    (
        [
            mk(
                Vec3::new(-corridor_half + left_w * 0.5, wall_y * 0.5, wall_z),
                Vec3::new(left_w.max(0.01), wall_y, wall_half_z * 2.0),
                10,
            ),
            mk(
                Vec3::new(gap_x + gap_half + right_w * 0.5, wall_y * 0.5, wall_z),
                Vec3::new(right_w.max(0.01), wall_y, wall_half_z * 2.0),
                11,
            ),
        ],
        gap_x,
    )
}



/// Dense buildings: packed boxes of varying height/width outside the
/// flight corridor, forming a skyline.
pub fn buildings(seed: u64) -> Vec<Prim> {
    let mut out = Vec::new();
    let n = 14 + (rng(seed, 50) * 8.0) as usize; // 14-21 buildings
    for i in 0..n {
        let b = (i * 7 + 60) as u64;
        let side = if rng(seed, b) > 0.5 { 1.0 } else { -1.0 };
        let h = 1.0 + rng(seed, b + 3) * 12.0;
        let w = 0.6 + rng(seed, b + 4) * 4.0;
        let x = side * (5.0 + rng(seed, b + 1) * 4.0);
        let z = -0.5 - rng(seed, b + 2) * 9.0;

        let d = w * (0.5 + rng(seed, b + 5));
        out.push(Prim {
            pos: Vec3::new(x, h * 0.5, z),
            scale: Vec3::new(w, h, d),
            rot: Quat::IDENTITY,
            color: color(seed, b + 6),
            pattern: pattern(seed, b + 7),
            mesh: MeshKind::Cube,
            aux: [0.0; 4],
        });
    }
    out
}

/// Weird branchy structure: recursive thin cylinders, no leaves.
/// Abstract branching — more like coral/antennae than trees.
pub fn tree(seed: u64, base: Vec3) -> Vec<Prim> {
    let mut out = Vec::new();
    let trunk_h = 2.5 + rng(seed, 70) * 5.0;
    let trunk_r = 0.03 + rng(seed, 71) * 0.08;
    let col = color(seed, 72);
    out.push(Prim {
        pos: base + Vec3::new(0.0, trunk_h * 0.5, 0.0),
        scale: Vec3::new(trunk_r, trunk_h, trunk_r),
        rot: Quat::IDENTITY,
        color: col,
        pattern: [0.0; 4],
        mesh: MeshKind::Cylinder,
        aux: [0.0; 4],
    });
    // recursive branches — noise-driven angles, sparse, organic
    let mut stack = vec![(base + Vec3::new(0.0, trunk_h, 0.0), trunk_h * 0.5, 0u32)];
    while let Some((pos, len, depth)) = stack.pop() {
        if depth >= 3 || len < 0.15 {
            continue;
        }
        // 1-2 children, noise-driven direction
        let n_children = 1 + (rng(seed, 75 + depth as u64) * 2.0) as usize;
        for c in 0..n_children {
            let ci = (depth * 10 + c as u32) as u64;
            // noise-based angle: smooth variation, not uniform random
            let angle = rng(seed, 76 + ci) * std::f32::consts::TAU
                + (depth as f32) * 0.8;
            let tilt = 0.2 + rng(seed, 77 + ci) * 0.5;
            let child_len = len * (0.6 + rng(seed, 78 + ci) * 0.3);
            let dir = Vec3::new(
                angle.cos() * tilt.sin(),
                tilt.cos(),
                angle.sin() * tilt.sin(),
            );
            let end = pos + dir * child_len;
            let r = trunk_r * (0.6 - depth as f32 * 0.15).max(0.1);
            out.push(Prim {
                pos: (pos + end) * 0.5,
                scale: Vec3::new(r, child_len, r),
                rot: Quat::from_rotation_arc(Vec3::Y, dir.normalize()),
                color: col,
                pattern: [0.0; 4],
                mesh: MeshKind::Cylinder,
                aux: [0.0; 4],
            });
            stack.push((end, child_len, depth + 1));
        }
    }
    out
}

/// Wire/fence: thin cylinders strung between points, forming a fence
/// line or overhead wires.
pub fn wires(seed: u64) -> Vec<Prim> {
    let mut out = Vec::new();
    let n = 8 + (rng(seed, 100) * 8.0) as usize; // 8-15 wire segments
    let mut prev = Vec3::new(
        (rng(seed, 101) - 0.5) * 8.0,
        0.8 + rng(seed, 102) * 1.5,
        -rng(seed, 103) * 8.0,
    );
    for i in 0..n {
        let b = (i * 5 + 110) as u64;
        let next = prev
            + Vec3::new(
                (rng(seed, b) - 0.5) * 3.0,
                (rng(seed, b + 1) - 0.5) * 0.5,
                -1.0 - rng(seed, b + 2) * 2.0,
            );
        let mid = (prev + next) * 0.5;
        let len = (next - prev).length();
        let r = 0.01 + rng(seed, b + 3) * 0.02;
        out.push(Prim {
            pos: mid,
            scale: Vec3::new(r, len, r),
            rot: Quat::IDENTITY,
            color: [0.15, 0.15, 0.15, 1.0],
            pattern: [0.0; 4],
            mesh: MeshKind::Cylinder,
            aux: [0.0; 4],
        });
        prev = next;
    }
    out
}

/// Ground cover: small boxes/cylinders scattered densely on the floor.
pub fn ground_cover(seed: u64) -> Vec<Prim> {
    let mut out = Vec::new();
    let n = 24 + (rng(seed, 130) * 16.0) as usize; // 24-39 items
    for i in 0..n {
        let b = (i * 6 + 140) as u64;
        let x = (rng(seed, b) - 0.5) * 8.5;
        let z = -rng(seed, b + 1) * 9.0;
        let h = 0.05 + rng(seed, b + 2) * 0.25;
        let w = 0.1 + rng(seed, b + 3) * 0.4;
        let is_cyl = rng(seed, b + 4) > 0.6;
        out.push(Prim {
            pos: Vec3::new(x, h * 0.5, z),
            scale: Vec3::new(w, h, w * (0.5 + rng(seed, b + 5))),
            rot: Quat::IDENTITY,
            color: color(seed, b + 6),
            pattern: pattern(seed, b + 7),
            mesh: if is_cyl { MeshKind::Cylinder } else { MeshKind::Cube },
            aux: [0.0; 4],
        });
    }
    out
}

/// Full scene: wall + buildings + trees + wires + ground cover.
/// Returns (prims, gap_x).
pub fn scene(seed: u64) -> (Vec<Prim>, f32) {
    let (wall_prims, gap_x) = wall(seed);
    let mut prims = Vec::with_capacity(64);
    prims.extend_from_slice(&wall_prims);
    prims.extend(buildings(seed));

    prims.extend(wires(seed));
    prims.extend(ground_cover(seed));
    (prims, gap_x)
}
