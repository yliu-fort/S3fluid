// Combines RK4 stages: z_next = z + dt/6 * (k1 + 2k2 + 2k3 + k4)
@group(0) @binding(0) var<storage, read> z_in: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> k1_in: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> k2_in: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> k3_in: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> k4_in: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> z_out: array<vec2<f32>>;

struct Config {
    nlat: u32,
    nlon: u32,
    lmax: u32,
    pad: u32
}

struct RK4Config {
    dt: f32,
    coeff: f32,
    pad1: f32,
    pad2: f32
}

@group(1) @binding(0) var<uniform> config: Config;
@group(1) @binding(1) var<uniform> rk4_config: RK4Config;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let l = global_id.x;
    let m = global_id.y;
    let M = config.lmax + 1u;

    if (l > config.lmax || m > config.lmax) {
        return;
    }

    let idx = m * M + l;

    if (l < m) {
        z_out[idx] = vec2<f32>(0.0, 0.0);
        return;
    }

    let z_val = z_in[idx];
    let k1 = k1_in[idx];
    let k2 = k2_in[idx];
    let k3 = k3_in[idx];
    let k4 = k4_in[idx];

    let multiplier = rk4_config.dt / 6.0;

    z_out[idx] = vec2<f32>(
        z_val.x + multiplier * (k1.x + 2.0 * k2.x + 2.0 * k3.x + k4.x),
        z_val.y + multiplier * (k1.y + 2.0 * k2.y + 2.0 * k3.y + k4.y)
    );
}
