// Legendre Synthesis for Theta Derivative
// Maps a(m, l) -> dtheta_freq(j, m)
// Uses dP_lm_dtheta instead of P_lm

@group(0) @binding(0) var<storage, read> coeff_in: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> freq_out: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> dP_lm_dtheta: array<f32>;

struct Config {
    nlat: u32,
    nlon: u32,
    lmax: u32,
    pad: u32
}

@group(1) @binding(0) var<uniform> config: Config;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let j = global_id.x;
    let m = global_id.y;
    let M = config.lmax + 1u;
    let L = config.lmax + 1u;

    if (j >= config.nlat || m > config.lmax) {
        return;
    }

    var sum_re = 0.0;
    var sum_im = 0.0;

    for (var l = m; l < L; l = l + 1u) {
        let ml_idx = m * L + l;
        let p_idx = j * (M * L) + ml_idx;
        let p_val = dP_lm_dtheta[p_idx];

        let c_val = coeff_in[ml_idx];

        sum_re = sum_re + p_val * c_val.x;
        sum_im = sum_im + p_val * c_val.y;
    }

    let freq_idx = j * M + m;
    let nlon_f32 = f32(config.nlon);
    freq_out[freq_idx] = vec2<f32>(sum_re * nlon_f32, sum_im * nlon_f32);
}
