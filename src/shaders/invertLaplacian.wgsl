@group(0) @binding(0) var<storage, read> in_a: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> lapEigs: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_a: array<vec2<f32>>;

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

    if (l > config.lmax || m > config.lmax) {
        return;
    }

    let idx = m * M + l;

    if (l < m || (l == 0u && m == 0u)) {
        out_a[idx] = vec2<f32>(0.0, 0.0);
        return;
    }

    let val = in_a[idx];
    let eig = lapEigs[idx];

    // eig is -l*(l+1). For valid m<=l and l!=0, eig is non-zero
    let inv_eig = 1.0 / eig;
    out_a[idx] = vec2<f32>(val.x * inv_eig, val.y * inv_eig);
}
