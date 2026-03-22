// Legendre Analysis
// Maps F(j, m) -> a(m, l)
//
// Workgroup setup:
// x: l index
// y: m index
// z: 1

@group(0) @binding(0) var<storage, read> freq_in: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> coeff_out: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> P_lm: array<f32>;

struct Config {
    nlat: u32,
    nlon: u32,
    lmax: u32,
    pad: u32
}

@group(1) @binding(0) var<uniform> config: Config;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let l = global_id.x;
    let m = global_id.y;
    let M = config.lmax + 1u;
    let L = config.lmax + 1u;

    if (l > config.lmax || m > config.lmax) {
        return;
    }

    let out_idx = m * L + l;

    if (l < m) {
        coeff_out[out_idx] = vec2<f32>(0.0, 0.0);
        return;
    }

    var sum_re = 0.0;
    var sum_im = 0.0;

    for (var j = 0u; j < config.nlat; j = j + 1u) {
        let p_idx = j * (M * L) + out_idx;
        let p_val = P_lm[p_idx];
        let p_w = p_val * w[j];

        let freq_val = freq_in[j * M + m];

        sum_re = sum_re + p_w * freq_val.x;
        sum_im = sum_im + p_w * freq_val.y;
    }

    // Include 2pi / nlon scaling (nlon was scaled in fft but needs exact 2pi/nlon for exact weight according to cpu reference)
    // CPU ref: sum * (2pi / nlon)
    let factor = 2.0 * 3.14159265358979323846 / f32(config.nlon);
    coeff_out[out_idx] = vec2<f32>(sum_re * factor, sum_im * factor);
}
