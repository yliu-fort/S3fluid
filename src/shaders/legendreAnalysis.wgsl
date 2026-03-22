@group(0) @binding(0) var<storage, read> freqIn: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> w_j: array<f32>;
@group(0) @binding(2) var<storage, read> P_lm: array<f32>;
@group(0) @binding(3) var<storage, read_write> aOut: array<vec2<f32>>;

struct Params {
    nlat: u32,
    nlon: u32,
    M: u32,
    L: u32
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let m = global_id.x;
    let l = global_id.y;

    if (m >= params.M || l >= params.L) {
        return;
    }

    if (l < m) {
        aOut[m * params.L + l] = vec2<f32>(0.0, 0.0);
        return;
    }

    var sum_re = 0.0;
    var sum_im = 0.0;

    let pi = 3.14159265358979323846;

    for (var j: u32 = 0; j < params.nlat; j++) {
        let weight = w_j[j];
        let p_val = P_lm[j * (params.M * params.L) + m * params.L + l];
        let f_val = freqIn[j * params.M + m];

        sum_re += weight * p_val * f_val.x;
        sum_im += weight * p_val * f_val.y;
    }

    // Multiply by 2pi / nlon
    let factor = 2.0 * pi / f32(params.nlon);
    aOut[m * params.L + l] = vec2<f32>(sum_re * factor, sum_im * factor);
}