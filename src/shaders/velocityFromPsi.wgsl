@group(0) @binding(0) var<storage, read> dpsiDphiGrid: array<f32>;
@group(0) @binding(1) var<storage, read> dpsiDthetaGrid: array<f32>;
@group(0) @binding(2) var<storage, read> sinTheta: array<f32>; // Length nlat
@group(0) @binding(3) var<storage, read_write> uThetaGrid: array<f32>;
@group(0) @binding(4) var<storage, read_write> uPhiGrid: array<f32>;

struct Params {
    nlat: u32,
    nlon: u32
}
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let lat = global_id.x;
    let lon = global_id.y;

    if (lat >= params.nlat || lon >= params.nlon) {
        return;
    }

    let idx = lat * params.nlon + lon;

    let sin_val = sinTheta[lat];

    uThetaGrid[idx] = dpsiDphiGrid[idx] / sin_val;
    uPhiGrid[idx] = -dpsiDthetaGrid[idx];
}
