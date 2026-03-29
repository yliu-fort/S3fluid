@group(0) @binding(0) var<storage, read> psiGrid: array<f32>;
@group(0) @binding(1) var<storage, read> zetaGrid: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> energyTerms: array<f32>;

struct Config {
    nlat: u32,
    nlon: u32,
    lmax: u32,
    pad: u32
}

@group(1) @binding(0) var<uniform> config: Config;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let k = global_id.x;
    let j = global_id.y;

    if (k >= config.nlon || j >= config.nlat) {
        return;
    }

    let out_idx = j * config.nlon + k;

    let psi = psiGrid[out_idx];
    let zeta = zetaGrid[out_idx];
    let w_val = w[j];

    // local energy term = -0.5 * psi * zeta * w_j * (2pi / nlon)
    let factor = 2.0 * 3.14159265358979323846 / f32(config.nlon);

    energyTerms[out_idx] = -0.5 * psi * zeta * w_val * factor;
}
