@group(0) @binding(0) var<storage, read> advLM: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> zetaLM: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> lapEigs: array<f32>;
@group(0) @binding(3) var<storage, read_write> rhsLM: array<vec2<f32>>;

struct Params {
    M: u32,
    L: u32,
    nu: f32
}
@group(0) @binding(4) var<uniform> params: Params;

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
        rhsLM[idx] = vec2<f32>(0.0, 0.0);
        return;
    }

    let adv = advLM[idx];
    let zeta = zetaLM[idx];
    let eig = lapEigs[idx];
    let nu = params.nu;

    // rhs = -advLM + nu * laplacian(zetaLM)
    let diff_x = nu * zeta.x * eig;
    let diff_y = nu * zeta.y * eig;

    rhsLM[idx] = vec2<f32>(-adv.x + diff_x, -adv.y + diff_y);
}
