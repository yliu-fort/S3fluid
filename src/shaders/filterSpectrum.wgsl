@group(0) @binding(0) var<storage, read> a_in: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> specFilter: array<f32>; // Length L
@group(0) @binding(2) var<storage, read_write> a_out: array<vec2<f32>>;

struct Params {
    nlat: u32,
    nlon: u32,
    M: u32,
    L: u32
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let m = global_id.x;
    let l = global_id.y;
    let M = params.M;
    let L = params.L;

    if (m >= M || l >= L) {
        return;
    }

    let idx = m * L + l;

    if (l < m) {
        a_out[idx] = vec2<f32>(0.0, 0.0);
        return;
    }

    let val = a_in[idx];
    let filt = specFilter[l];

    a_out[idx] = vec2<f32>(val.x * filt, val.y * filt);
}
