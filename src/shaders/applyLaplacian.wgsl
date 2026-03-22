struct Params {
    nlat: u32,
    nlon: u32,
    M: u32,
    L: u32,
}

@group(0) @binding(0) var<storage, read> aIn : array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> lapEigs : array<f32>;
@group(0) @binding(2) var<storage, read_write> aOut : array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params : Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let m = global_id.x;
    let l = global_id.y;

    if (m >= params.M || l >= params.L) {
        return;
    }

    let idx = m * params.L + l;

    // Default to zero
    aOut[idx] = vec2<f32>(0.0, 0.0);

    if (l >= m) {
        let a = aIn[idx];
        let eig = lapEigs[idx];

        // (a.x + i * a.y) * eig = (a.x * eig) + i * (a.y * eig)
        aOut[idx] = vec2<f32>(a.x * eig, a.y * eig);
    }
}
