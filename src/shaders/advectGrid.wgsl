@group(0) @binding(0) var<storage, read> uThetaGrid: array<f32>;
@group(0) @binding(1) var<storage, read> uPhiGrid: array<f32>;
@group(0) @binding(2) var<storage, read> dzetaDthetaGrid: array<f32>;
@group(0) @binding(3) var<storage, read> dzetaDphiGrid: array<f32>;
@group(0) @binding(4) var<storage, read> sinTheta: array<f32>;
@group(0) @binding(5) var<storage, read_write> advGrid: array<f32>;

struct Params {
    nlat: u32,
    nlon: u32
}
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let lat = global_id.x;
    let lon = global_id.y;

    if (lat >= params.nlat || lon >= params.nlon) {
        return;
    }

    let idx = lat * params.nlon + lon;

    let uTheta = uThetaGrid[idx];
    let uPhi = uPhiGrid[idx];
    let dzetaDtheta = dzetaDthetaGrid[idx];
    let dzetaDphi = dzetaDphiGrid[idx];
    let sin_val = sinTheta[lat];

    // adv = u_theta * dzeta_dtheta + u_phi * dzeta_dphi / sinTheta
    advGrid[idx] = uTheta * dzetaDtheta + uPhi * (dzetaDphi / sin_val);
}
