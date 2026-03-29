// Random Grid Initialization
// Maps global_id -> grid(j, k)
//
// A simple hash function to generate deterministic pseudo-random noise
// based on the seed and spatial coordinates.

@group(0) @binding(0) var<storage, read_write> grid_out: array<f32>;

struct Config {
    nlat: u32,
    nlon: u32,
    lmax: u32,
    pad: u32
}

struct InitConfig {
    seed: f32,
    amplitude: f32
}

@group(1) @binding(0) var<uniform> config: Config;
@group(1) @binding(1) var<uniform> init_config: InitConfig;

// A simple pseudo-random hash
fn hash(val: vec3<f32>) -> f32 {
    let p3 = fract(vec3<f32>(val.xyx) * 0.1031);
    let p3_2 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3_2.x + p3_2.y) * p3_2.z);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let k = global_id.x;
    let j = global_id.y;

    if (k >= config.nlon || j >= config.nlat) {
        return;
    }

    let out_idx = j * config.nlon + k;

    // value in [-1, 1]
    let noise = hash(vec3<f32>(f32(k), f32(j), init_config.seed)) * 2.0 - 1.0;

    // We apply amplitude directly here before analysis
    grid_out[out_idx] = noise * init_config.amplitude;
}
