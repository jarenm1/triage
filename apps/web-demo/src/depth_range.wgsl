struct Range {
    minimum: atomic<u32>,
    maximum: atomic<u32>,
}
@group(0) @binding(0) var depth: texture_2d<f32>;
@group(0) @binding(1) var ids: texture_2d<u32>;
@group(0) @binding(2) var<storage, read_write> range: Range;
var<workgroup> local_minimum: atomic<u32>;
var<workgroup> local_maximum: atomic<u32>;

@compute @workgroup_size(16, 16)
fn reduce(@builtin(global_invocation_id) pixel: vec3<u32>, @builtin(local_invocation_index) index: u32) {
    if index == 0u {
        atomicStore(&local_minimum, 0x7f800000u);
        atomicStore(&local_maximum, 0u);
    }
    workgroupBarrier();
    if all(pixel.xy < textureDimensions(depth)) {
        let p = vec2<i32>(pixel.xy);
        let d = textureLoad(depth, p, 0).r;
        // Positive finite float bits preserve numeric order. Background is not geometry.
        if textureLoad(ids, p, 0).r != 0u && d > 0. && d <= 3.402823e38 {
            atomicMin(&local_minimum, bitcast<u32>(d));
            atomicMax(&local_maximum, bitcast<u32>(d));
        }
    }
    workgroupBarrier();
    if index == 0u {
        let minimum = atomicLoad(&local_minimum);
        let maximum = atomicLoad(&local_maximum);
        if minimum <= maximum {
            atomicMin(&range.minimum, minimum);
            atomicMax(&range.maximum, maximum);
        }
    }
}
