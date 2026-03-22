@group(0) @binding(0) var<storage, read> F_m: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> P_lm: array<f32>;
@group(0) @binding(3) var<storage, read_write> a_lm: array<vec2<f32>>;

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

    let M = params.M;
    let L = params.L;
    let nlat = params.nlat;
    let nlon = params.nlon;

    if (m >= M || l >= L) {
        return;
    }

    let ml_idx = m * L + l;

    if (l < m) {
        a_lm[ml_idx] = vec2<f32>(0.0, 0.0);
        return;
    }

    var sum_re = 0.0;
    var sum_im = 0.0;

    for (var j: u32 = 0; j < nlat; j++) {
        let p_val = P_lm[j * (M * L) + ml_idx];
        let weight = w[j];

        let p_w = p_val * weight;

        let f = F_m[j * M + m];

        sum_re += p_w * f.x;
        sum_im += p_w * f.y;
    }

    let factor = 2.0 * 3.14159265358979323846 / f32(nlon);
    a_lm[ml_idx] = vec2<f32>(sum_re * factor, sum_im * factor);
}
