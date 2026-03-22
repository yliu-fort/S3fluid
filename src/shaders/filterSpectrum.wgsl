@group(0) @binding(0) var<storage, read_write> a: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> specFilter: array<f32>; // size L
@group(0) @binding(2) var<uniform> uniforms: vec2<u32>; // (L, M)

// filterSpectrum
// a(m, l) *= specFilter(l)

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

    let filter_val = specFilter[l];

    let val = a[index];
    a[index] = vec2<f32>(val.x * filter_val, val.y * filter_val);
}
