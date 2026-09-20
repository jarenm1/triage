struct Globals {
    view: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    inv_view_projection: mat4x4<f32>,
    camera_position: vec4<f32>,
    light_direction: vec4<f32>,
    light_color: vec4<f32>,
    ambient_color: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> globals: Globals;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) model_0: vec4<f32>,
    @location(3) model_1: vec4<f32>,
    @location(4) model_2: vec4<f32>,
    @location(5) model_3: vec4<f32>,
    @location(6) color: vec4<f32>,
    @location(7) object_id: u32,
    @location(8) pattern: vec4<f32>,
    @location(9) aux: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) view_depth: f32,
    @location(3) @interpolate(flat) object_id: u32,
    @location(4) world_position: vec3<f32>,
    @location(5) @interpolate(flat) pattern: vec4<f32>,
    @location(6) @interpolate(flat) aux: vec4<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let model = mat4x4<f32>(input.model_0, input.model_1, input.model_2, input.model_3);
    var world_position = model * vec4<f32>(input.position, 1.0);
    // terrain: displace Y by world-space noise (pattern.x == 9)
    // objects: snap to terrain at base xz (aux.x > 0.5, base in aux.yz)
    let kind = u32(input.pattern.x + 0.5);
    if kind == 9u {
        world_position.y = terrain_height(world_position.xz);
    } else if input.aux.x > 0.5 {
        let base_xz = vec2<f32>(input.aux.y, input.aux.z);
        world_position.y += terrain_height(base_xz);
    }
    var output: VertexOutput;
    output.clip_position = globals.view_projection * world_position;
    output.world_normal = normalize((model * vec4<f32>(input.normal, 0.0)).xyz);
    output.color = input.color;
    output.view_depth = -(globals.view * world_position).z;
    output.object_id = input.object_id;
    output.pattern = input.pattern;
    output.aux = input.aux;
    output.world_position = world_position.xyz;
    return output;
}
fn hash3(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.3183099 + vec3<f32>(0.1, 0.2, 0.3));
    q = q * 17.0;
    return fract(q.x * q.y * q.z * (q.x + q.y + q.z));
}

fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = hash3(i);
    let n100 = hash3(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash3(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash3(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash3(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash3(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash3(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash3(i + vec3<f32>(1.0, 1.0, 1.0));
    let nx00 = mix(n000, n100, u.x);
    let nx10 = mix(n010, n110, u.x);
    let nx01 = mix(n001, n101, u.x);
    let nx11 = mix(n011, n111, u.x);
    return mix(mix(nx00, nx10, u.y), mix(nx01, nx11, u.y), u.z);
}

fn voronoi(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    var min_dist = 8.0;
    for (var x = -1; x <= 1; x++) {
        for (var y = -1; y <= 1; y++) {
            for (var z = -1; z <= 1; z++) {
                let cell = i + vec3<f32>(f32(x), f32(y), f32(z));
                let point = vec3<f32>(
                    hash3(cell),
                    hash3(cell + vec3<f32>(1.0, 0.0, 0.0)),
                    hash3(cell + vec3<f32>(0.0, 0.0, 1.0)),
                );
                let d = length(f - vec3<f32>(f32(x), f32(y), f32(z)) - point);
                min_dist = min(min_dist, d);
            }
        }
    }
    return min_dist;
}

fn terrain_height(xz: vec2<f32>) -> f32 {
    // world-space terrain: continuous across chunk boundaries
    let p = vec3<f32>(xz.x, 0.0, xz.y);
    return value_noise(p * 0.02) * 6.0
         + value_noise(p * 0.08) * 2.0
         + value_noise(p * 0.3) * 0.5;
}

fn pattern_factor(input: VertexOutput) -> f32 {
    let kind = u32(input.pattern.x + 0.5);
    if kind == 0u {
        return 1.0;
    }
    let scale = max(input.pattern.y, 0.01);
    let seed = input.pattern.z;
    let contrast = input.pattern.w;
    let p = input.world_position * scale + vec3<f32>(seed, seed * 1.7, seed * 2.3);
    var v = 0.5;
    if kind == 1u {
        let c = floor(p.x) + floor(p.y) + floor(p.z);
        v = select(0.0, 1.0, (c - floor(c * 0.5) * 2.0) > 0.5);
    } else if kind == 2u {
        v = select(0.0, 1.0, fract(p.x) > 0.5);
    } else if kind == 3u {
        v = value_noise(p) * 0.7 + value_noise(p * 2.7) * 0.3;
    } else if kind == 4u {
        v = fract(p.x);
    } else if kind == 5u {
        // voronoi cells
        v = voronoi(p);
    } else if kind == 6u {
        // brick: offset rows
        let row = floor(p.y);
        let offset = select(0.0, 0.5, (row - floor(row * 0.5) * 2.0) > 0.5);
        let bx = fract(p.x + offset);
        let by = fract(p.y);
        v = select(0.0, 1.0, bx > 0.08 && by > 0.08);
    } else if kind == 7u {
        // grid lines
        let gx = abs(fract(p.x) - 0.5);
        let gz = abs(fract(p.z) - 0.5);
        v = select(1.0, 0.0, gx < 0.05 || gz < 0.05);
    } else if kind == 8u {
        // concentric rings
        v = select(0.0, 1.0, fract(length(p.xz)) > 0.5);
    } else if kind == 9u {
        // terrain: multi-octave value noise, world-space
        v = value_noise(p * 0.5) * 0.5
          + value_noise(p * 2.0) * 0.3
          + value_noise(p * 8.0) * 0.2;
    }
    return mix(1.0, v, clamp(contrast, 0.0, 1.0));
}

fn lit_color(input: VertexOutput) -> vec4<f32> {
    let N = normalize(input.world_normal);
    let L = normalize(globals.light_direction.xyz);
    let V = normalize(globals.camera_position.xyz - input.world_position);
    let H = normalize(L + V);

    // Primary directional diffuse
    let NdotL = max(dot(N, L), 0.0);
    let diffuse = globals.light_color.rgb * NdotL;

    // Specular highlight (Blinn-Phong)
    let NdotH = max(dot(N, H), 0.0);
    let specular_power = 32.0;
    let specular_factor = pow(NdotH, specular_power) * select(0.0, 1.0, NdotL > 0.0);
    let specular = globals.light_color.rgb * (specular_factor * 0.6);

    // Secondary fill / ground bounce light
    let fill_dir = normalize(vec3<f32>(-L.x * 0.7, 0.3, -L.z * 0.7));
    let fill_diffuse = max(dot(N, fill_dir), 0.0) * vec3<f32>(0.28, 0.32, 0.38);

    // Ambient light
    let ambient = globals.ambient_color.rgb;

    let lighting = ambient + diffuse + fill_diffuse;
    let base = input.color.rgb * pattern_factor(input);
    let final_rgb = base * lighting + specular;
    return vec4<f32>(final_rgb, input.color.a);
}

@fragment
fn fs_display(input: VertexOutput) -> @location(0) vec4<f32> {
    return lit_color(input);
}

struct SensorOutput {
    @location(0) color: vec4<f32>,
    @location(1) depth: f32,
    @location(2) object_id: u32,
}

@fragment
fn fs_sensor(input: VertexOutput) -> SensorOutput {
    var output: SensorOutput;
    output.color = lit_color(input);
    output.depth = input.view_depth;
    output.object_id = input.object_id;
    return output;
}

@group(1) @binding(0)
var skybox_texture: texture_2d<f32>;

@group(1) @binding(1)
var skybox_sampler: sampler;

struct SkyboxOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ray_dir: vec3<f32>,
}

@vertex
fn vs_skybox(@builtin(vertex_index) vertex_index: u32) -> SkyboxOutput {
    var output: SkyboxOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32((vertex_index >> 1u) & 1u) * 4 - 1);
    let clip = vec4<f32>(x, y, 1.0, 1.0);
    output.clip_position = clip;
    let world = globals.inv_view_projection * clip;
    output.ray_dir = (world.xyz / world.w) - globals.camera_position.xyz;
    return output;
}

@fragment
fn fs_skybox(input: SkyboxOutput) -> @location(0) vec4<f32> {
    let d = normalize(input.ray_dir);
    let phi = atan2(d.z, d.x);
    let theta = asin(clamp(d.y, -1.0, 1.0));
    let u = phi / 6.28318530718 + 0.5;
    let v = 0.5 - theta / 3.14159265359;
    return textureSample(skybox_texture, skybox_sampler, vec2<f32>(u, v));
}
