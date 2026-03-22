@group(0) @binding(0) var<storage, read> uThetaGrid: array<f32>;
@group(0) @binding(1) var<storage, read> uPhiGrid: array<f32>;
@group(0) @binding(2) var<storage, read> dzetaDthetaGrid: array<f32>;
@group(0) @binding(3) var<storage, read> dzetaDphiGrid: array<f32>;
@group(0) @binding(4) var<storage, read> sinTheta: array<f32>;
@group(0) @binding(5) var<storage, read_write> advGrid: array<f32>;

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

    let u_theta = uThetaGrid[out_idx];
    let u_phi = uPhiGrid[out_idx];

    let dzeta_dtheta = dzetaDthetaGrid[out_idx];

    // dzetaDphi is just raw dzeta/dphi from SHT synthesis, we must divide by sinTheta here
    let inv_sin = 1.0 / sinTheta[j];
    let dzeta_dphi = dzetaDphiGrid[out_idx] * inv_sin;

    // advection = u_theta * dzeta/dtheta + u_phi * dzeta/dphi
    let adv = u_theta * dzeta_dtheta + u_phi * dzeta_dphi;

    advGrid[out_idx] = adv;
}
