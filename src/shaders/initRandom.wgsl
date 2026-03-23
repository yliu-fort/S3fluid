@group(0) @binding(0) var<storage, read_write> grid_out: array<f32>;

struct Params {
    nlat: u32,
    nlon: u32,
    seed: u32,
}
@group(0) @binding(1) var<uniform> params: Params;

// A simple hash function
fn hash(s: u32) -> f32 {
    var x = s;
    x = x ^ (x >> 16);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16);
    return f32(x) / f32(0xffffffffu);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let lat = global_id.x;
    let lon = global_id.y;

    if (lat >= params.nlat || lon >= params.nlon) {
        return;
    }

    let idx = lat * params.nlon + lon;

    let cell_seed = params.seed ^ (lat * 1973u + lon * 9277u);
    let rand_val = hash(cell_seed) * 2.0 - 1.0;

    grid_out[idx] = rand_val;
}
