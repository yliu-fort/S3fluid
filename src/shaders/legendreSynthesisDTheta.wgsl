@group(0) @binding(0) var<storage, read> aIn: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> dP_lm_dtheta: array<f32>;
@group(0) @binding(2) var<storage, read_write> freqOut: array<vec2<f32>>;

struct Params {
    nlat: u32,
    nlon: u32,
    M: u32,
    L: u32
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let m = global_id.x;
    let j = global_id.y;

    if (m >= params.M || j >= params.nlat) {
        return;
    }

    var sum_re = 0.0;
    var sum_im = 0.0;

    for (var l: u32 = 0; l < params.L; l++) {
        if (l < m) {
            continue;
        }

        let a_val = aIn[m * params.L + l];
        let dp_val = dP_lm_dtheta[j * (params.M * params.L) + m * params.L + l];

        sum_re += a_val.x * dp_val;
        sum_im += a_val.y * dp_val;
    }

    let factor = f32(params.nlon);
    freqOut[j * params.M + m] = vec2<f32>(sum_re * factor, sum_im * factor);
}