@group(0) @binding(0) var<storage, read_write> a: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> lapEigs: array<f32>; // size L*M, flattened
@group(0) @binding(2) var<uniform> uniforms: vec2<u32>; // (L, M)

// invertLaplacian
// a(m, l) /= lapEigs(m, l)
// lapEigs(m, l) = -l*(l+1)
// Special case: if l == 0 (and therefore m == 0), set to zero.

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let L = uniforms.x;
    let M = uniforms.y;

    let total_elements = L * M;
    if (index >= total_elements) {
        return;
    }

    let m = index / L;
    let l = index % L;

    if (l < m) {
        return;
    }

    if (l == 0u) {
        a[index] = vec2<f32>(0.0, 0.0);
        return;
    }

    let eig = lapEigs[index];

    let val = a[index];
    a[index] = vec2<f32>(val.x / eig, val.y / eig);
}
