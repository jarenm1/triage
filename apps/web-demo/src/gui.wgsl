struct Globals {
    // Backing-pixel canvas dimensions, surface sRGB encoding flag, padding.
    size_transfer: vec4<f32>,
}
@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var atlas: texture_2d<f32>;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) color: vec4<f32>,
    @location(2) @interpolate(flat) glyph: f32,
}

@vertex fn vs_main(
    @builtin(vertex_index) vertex: u32,
    @location(0) rect: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) glyph: f32,
) -> VertexOut {
    let corners = array<vec2<f32>, 6>(
        vec2(0., 0.), vec2(1., 0.), vec2(0., 1.),
        vec2(0., 1.), vec2(1., 0.), vec2(1., 1.),
    );
    let uv = corners[vertex];
    let pixel = rect.xy + uv * rect.zw;
    var out: VertexOut;
    out.position = vec4(pixel / globals.size_transfer.xy * vec2(2., -2.) + vec2(-1., 1.), 0., 1.);
    out.uv = uv;
    out.color = color;
    out.glyph = glyph;
    return out;
}

fn linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + vec3(.055)) / 1.055, vec3(2.4)), c / 12.92, c <= vec3(.04045));
}

@fragment fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    var alpha = in.color.a;
    if in.glyph >= 0. {
        let glyph = u32(in.glyph);
        let origin = vec2<u32>(glyph % 16u, glyph / 16u) * 8u;
        let pixel = min(vec2<u32>(in.uv * 8.), vec2<u32>(7u));
        alpha *= textureLoad(atlas, vec2<i32>(origin + pixel), 0).r;
    }
    // Like the selector compositor, only linearize display colors when the
    // attachment itself encodes sRGB. Unorm surfaces receive display values.
    var color = in.color.rgb;
    if globals.size_transfer.z != 0. {
        color = linear(color);
    }
    return vec4(color, alpha);
}
