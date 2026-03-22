@group(0) @binding(0) var<storage, read> dpsiDphiGrid: array<f32>;
@group(0) @binding(1) var<storage, read> dpsiDthetaGrid: array<f32>;
@group(0) @binding(2) var<storage, read> sinTheta: array<f32>;
@group(0) @binding(3) var<storage, read_write> uThetaGrid: array<f32>;
@group(0) @binding(4) var<storage, read_write> uPhiGrid: array<f32>;

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
    let inv_sin = 1.0 / sinTheta[j];

    // uTheta = dpsi/dphi / sin(theta)
    uThetaGrid[out_idx] = dpsiDphiGrid[out_idx] * inv_sin;

    // uPhi = -dpsi/dtheta
    uPhiGrid[out_idx] = -dpsiDthetaGrid[out_idx];
}
