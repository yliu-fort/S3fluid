@group(0) @binding(0) var<storage, read_write> a: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> uniforms: vec2<u32>; // (L, M)

// (m, l)
// a is an array of vec2<f32>.
// We want to compute i * m * a(m,l)
// complex mult: (i * m) * (a_r + i a_i) = -m * a_i + i * m * a_r
// So real part becomes -m * a_i, imag part becomes m * a_r

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

    let fm = f32(m);

    let val = a[index];
    let new_real = -fm * val.y;
    let new_imag = fm * val.x;

    a[index] = vec2<f32>(new_real, new_imag);
}
