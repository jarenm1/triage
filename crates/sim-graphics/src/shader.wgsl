struct Globals {
    view: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    light_direction: vec4<f32>,
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
    return output;
}

fn lit_color(input: VertexOutput) -> vec4<f32> {
    let diffuse = max(dot(normalize(input.world_normal), normalize(globals.light_direction.xyz)), 0.0);
    let lighting = 0.18 + 0.82 * diffuse;
    return vec4<f32>(input.color.rgb * lighting, input.color.a);
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
