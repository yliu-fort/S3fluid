// Simple parallel reduction
// Performs a full reduction via strided loops for simplicity. Not the fastest,
// but enough for our grid sizes where nlon * nlat ~ 8192

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;

struct Config {
    count: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32
}

@group(1) @binding(0) var<uniform> config: Config;

// Maximum workgroup size is 256.
// We will launch just one workgroup of 256 threads to sum the entire array.
// Each thread handles multiple elements.

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tid = local_id.x;
    var thread_sum = 0.0;

    // Each thread sums a strided portion of the input array
    var gid = tid;
    while (gid < config.count) {
        thread_sum += input_data[gid];
        gid += 256u;
    }

    shared_data[tid] = thread_sum;

    workgroupBarrier();

    // standard tree reduction within the workgroup
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output_data[0] = shared_data[0];
    }
}
