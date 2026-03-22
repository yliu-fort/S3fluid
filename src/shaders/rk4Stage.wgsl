@group(0) @binding(0) var<storage, read> zetaLM: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> k: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> zetaTemp: array<vec2<f32>>;

struct Params {
    M: u32,
    L: u32,
    dt: f32,
    coeff: f32
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
        zetaTemp[idx] = vec2<f32>(0.0, 0.0);
        return;
    }

    let z = zetaLM[idx];
    let k_val = k[idx];

    let factor = params.dt * params.coeff;

    zetaTemp[idx] = vec2<f32>(z.x + factor * k_val.x, z.y + factor * k_val.y);
}
