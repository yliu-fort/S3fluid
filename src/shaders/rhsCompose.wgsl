@group(0) @binding(0) var<storage, read> advLM: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> zetaLM: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> lapEigs: array<f32>;
@group(0) @binding(3) var<storage, read_write> rhsLM: array<vec2<f32>>;

struct Config {
    nlat: u32,
    nlon: u32,
    lmax: u32,
    pad: u32
}

struct RHSConfig {
    nu: f32,
    pad1: f32,
    pad2: f32,
    pad3: f32
}

@group(1) @binding(0) var<uniform> config: Config;
@group(1) @binding(1) var<uniform> rhs_config: RHSConfig;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let l = global_id.x;
    let m = global_id.y;
    let M = config.lmax + 1u;

    if (l > config.lmax || m > config.lmax) {
        return;
    }

    let idx = m * M + l;

    if (l < m) {
        rhsLM[idx] = vec2<f32>(0.0, 0.0);
        return;
    }

    let adv_val = advLM[idx];
    let zeta_val = zetaLM[idx];
    let eig = lapEigs[idx];
    let nu = rhs_config.nu;

    // diff = nu * laplacian(zetaLM)
    let diff_re = nu * eig * zeta_val.x;
    let diff_im = nu * eig * zeta_val.y;

    // rhs = -adv_lm + diff_lm
    rhsLM[idx] = vec2<f32>(-adv_val.x + diff_re, -adv_val.y + diff_im);
}
