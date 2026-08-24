struct Globals {
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
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec4<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let model = mat4x4<f32>(input.model_0, input.model_1, input.model_2, input.model_3);
    var output: VertexOutput;
    output.clip_position = globals.view_projection * model * vec4<f32>(input.position, 1.0);
    output.world_normal = normalize((model * vec4<f32>(input.normal, 0.0)).xyz);
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let diffuse = max(dot(normalize(input.world_normal), normalize(globals.light_direction.xyz)), 0.0);
    let lighting = 0.18 + 0.82 * diffuse;
    return vec4<f32>(input.color.rgb * lighting, input.color.a);
}
