struct Globals {
    // Canvas dimensions, selected mode, whether the surface performs sRGB encoding.
    size_options: vec4<f32>,
}
@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var rgb: texture_2d<f32>;
@group(0) @binding(2) var depth: texture_2d<f32>;
@group(0) @binding(3) var ids: texture_2d<u32>;
struct DepthRange { minimum: u32, maximum: u32 }
@group(0) @binding(4) var<storage, read> depth_range: DepthRange;

@vertex fn vs_main(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(vec2(-1., -1.), vec2(3., -1.), vec2(-1., 3.));
    return vec4(positions[vertex], 0., 1.);
}
fn linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + vec3(.055)) / 1.055, vec3(2.4)), c / 12.92, c <= vec3(.04045));
}
fn srgb(c: vec3<f32>) -> vec3<f32> {
    return select(1.055 * pow(max(c, vec3(0.)), vec3(1. / 2.4)) - vec3(.055), 12.92 * c, c <= vec3(.0031308));
}
// Identical full-width uint32 hash and integer palette quantization to sim-inspection.
fn palette(id: u32) -> vec3<f32> {
    if id == 0u { return vec3(0.); }
    var h = id;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return vec3<f32>(vec3<u32>(64u) + (vec3<u32>(h, h >> 8u, h >> 16u) & vec3<u32>(255u)) * 191u / 255u) / 255.;
}
@fragment fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    var mode = u32(globals.size_options.z);
    var size = globals.size_options.xy;
    var pixel = position.xy;
    if mode == 3u {
        if size.x >= size.y {
            size.x /= 3.;
            mode = min(u32(pixel.x / size.x), 2u);
            pixel.x -= f32(mode) * size.x;
        } else {
            size.y /= 3.;
            mode = min(u32(pixel.y / size.y), 2u);
            pixel.y -= f32(mode) * size.y;
        }
    }
    let fitted = vec2(min(size.x, size.y * 4. / 3.), min(size.y, size.x * 3. / 4.));
    let uv = (pixel - (size - fitted) * .5) / fitted;
    if any(uv < vec2(0.)) || any(uv >= vec2(1.)) { return vec4(0., 0., 0., 1.); }
    let dimensions = textureDimensions(rgb);
    let p = vec2<i32>(min(vec2<u32>(uv * vec2<f32>(dimensions)), dimensions - vec2<u32>(1u)));
    var c = textureLoad(rgb, p, 0).rgb;
    if mode == 1u {
        let id = textureLoad(ids, p, 0).r;
        let d = textureLoad(depth, p, 0).r;
        let nearest = bitcast<f32>(depth_range.minimum);
        let farthest = bitcast<f32>(depth_range.maximum);
        // Auto-range visible geometry only. A constant-depth surface uses the midpoint.
        var t = .5;
        if farthest - nearest > 1e-5 {
            t = clamp((d - nearest) / (farthest - nearest), 0., 1.);
        }
        c = linear(vec3(1. - t, 1. - abs(2. * t - 1.), t));
        if id == 0u { c = vec3(0.); }
        else if !(d >= 0. || d < 0.) || abs(d) > 3.402823e38 { c = vec3(1., 0., 1.); }
    }
    if mode == 2u { c = linear(palette(textureLoad(ids, p, 0).r)); }
    if globals.size_options.w == 0. { c = srgb(c); }
    return vec4(c, 1.);
}
