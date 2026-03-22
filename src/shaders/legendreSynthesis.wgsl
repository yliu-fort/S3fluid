@group(0) @binding(0) var<storage, read> a_lm: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> P_lm: array<f32>;
@group(0) @binding(2) var<storage, read_write> freq_out: array<vec2<f32>>;

struct Params {
    nlat: u32,
    nlon: u32,
    M: u32,
    L: u32
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let j = global_id.x; // latitude
    let m = global_id.y; // wavenumber

    let M = params.M;
    let L = params.L;
    let nlat = params.nlat;

    if (j >= nlat || m >= M) {
        return;
    }

    var sum_re = 0.0;
    var sum_im = 0.0;

    let base_idx = j * (M * L) + m * L;

    // We only sum for l >= m
    for (var l: u32 = m; l < L; l++) {
        let p_val = P_lm[base_idx + l];
        let a = a_lm[m * L + l];

        sum_re += p_val * a.x;
        sum_im += p_val * a.y;
    }

    // Multiply by nlon since the IFFT later divides by nlon
    let factor = f32(params.nlon);
    freq_out[j * M + m] = vec2<f32>(sum_re * factor, sum_im * factor);
}
