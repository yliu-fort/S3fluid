@group(0) @binding(0) var<storage, read> gridIn: array<f32>;
@group(0) @binding(1) var<storage, read_write> freqOut: array<vec2<f32>>;

struct Params {
    nlat: u32,
    nlon: u32,
    M: u32,
    L: u32
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let j = global_id.x; // latitude index

    if (j >= params.nlat) {
        return;
    }

    let nlon = params.nlon;
    let M = params.M;
    let pi = 3.14159265358979323846;

    // Output is stored as freqOut[j * M + m]
    // O(N^2) DFT since N is small (e.g. 64)
    for (var m: u32 = 0; m < M; m++) {
        var sum_re = 0.0;
        var sum_im = 0.0;

        for (var k: u32 = 0; k < nlon; k++) {
            let val = gridIn[j * nlon + k];
            let angle = -2.0 * pi * f32(m) * f32(k) / f32(nlon);
            sum_re += val * cos(angle);
            sum_im += val * sin(angle);
        }

        freqOut[j * M + m] = vec2<f32>(sum_re, sum_im);
    }
}
