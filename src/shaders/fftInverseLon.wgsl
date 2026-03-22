@group(0) @binding(0) var<storage, read> freqIn: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> gridOut: array<f32>;

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

    for (var k: u32 = 0; k < nlon; k++) {
        var sum = 0.0;

        // For m = 0
        let f0 = freqIn[j * M + 0];
        sum += f0.x;

        // For m > 0 up to M-1
        for (var m: u32 = 1; m < M; m++) {
            let f = freqIn[j * M + m];
            let angle = 2.0 * pi * f32(m) * f32(k) / f32(nlon);
            sum += 2.0 * (f.x * cos(angle) - f.y * sin(angle));
        }

        gridOut[j * nlon + k] = sum / f32(nlon);
    }
}
