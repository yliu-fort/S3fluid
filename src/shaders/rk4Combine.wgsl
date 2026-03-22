@group(0) @binding(0) var<storage, read> zetaLM: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> k1: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> k2: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> k3: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> k4: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> specFilter: array<f32>;
@group(0) @binding(6) var<storage, read_write> zetaNext: array<vec2<f32>>;

struct Params {
    M: u32,
    L: u32,
    dt: f32
}
@group(0) @binding(7) var<uniform> params: Params;

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
        zetaNext[idx] = vec2<f32>(0.0, 0.0);
        return;
    }

    let z = zetaLM[idx];
    let k1_v = k1[idx];
    let k2_v = k2[idx];
    let k3_v = k3[idx];
    let k4_v = k4[idx];

    let filt = specFilter[l];
    let dt6 = params.dt / 6.0;

    let sum_x = k1_v.x + 2.0 * k2_v.x + 2.0 * k3_v.x + k4_v.x;
    let sum_y = k1_v.y + 2.0 * k2_v.y + 2.0 * k3_v.y + k4_v.y;

    let next_x = (z.x + dt6 * sum_x) * filt;
    let next_y = (z.y + dt6 * sum_y) * filt;

    zetaNext[idx] = vec2<f32>(next_x, next_y);
}
