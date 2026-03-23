@group(0) @binding(0) var<storage, read> psiGrid: array<f32>;
@group(0) @binding(1) var<storage, read> zetaGrid: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>; // Length nlat
@group(0) @binding(3) var<storage, read_write> energyTerms: array<f32>; // Length nlat*nlon

struct Params {
    nlat: u32,
    nlon: u32
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let lat = global_id.x;
    let lon = global_id.y;

    if (lat >= params.nlat || lon >= params.nlon) {
        return;
    }

    let idx = lat * params.nlon + lon;

    let psi = psiGrid[idx];
    let zeta = zetaGrid[idx];
    let weight = w[lat];

    // -0.5 * psi * zeta * weight
    energyTerms[idx] = -0.5 * psi * zeta * weight;
}
