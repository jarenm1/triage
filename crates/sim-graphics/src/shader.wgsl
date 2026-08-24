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
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) view_depth: f32,
    @location(3) @interpolate(flat) object_id: u32,
    @location(4) world_position: vec3<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let model = mat4x4<f32>(input.model_0, input.model_1, input.model_2, input.model_3);
    let world_position = model * vec4<f32>(input.position, 1.0);
    var output: VertexOutput;
    output.clip_position = globals.view_projection * world_position;
    output.world_normal = normalize((model * vec4<f32>(input.normal, 0.0)).xyz);
    output.color = input.color;
    output.view_depth = -(globals.view * world_position).z;
    output.object_id = input.object_id;
    output.world_position = world_position.xyz;
    return output;
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
    let final_rgb = input.color.rgb * lighting + specular;
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
