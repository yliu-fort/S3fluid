@group(0) @binding(0) var<storage, read> inArray: array<f32>;
@group(0) @binding(1) var<storage, read_write> outArray: array<f32>;

struct Params {
    count: u32
}
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> sharedData: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let gid = global_id.x;
    let lid = local_id.x;

    var sum = 0.0;

    // Each thread sums up elements with stride
    for (var i = gid; i < params.count; i += 256 * 64) {
        sum += inArray[i];
    }

    sharedData[lid] = sum;
    workgroupBarrier();

    // Reduce within workgroup
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) {
            sharedData[lid] += sharedData[lid + s];
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        outArray[workgroup_id.x] = sharedData[0];
    }
}
