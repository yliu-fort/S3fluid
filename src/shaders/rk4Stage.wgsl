// Computes an intermediate stage for RK4: z_temp = z + coeff * dt * k
@group(0) @binding(0) var<storage, read> z_in: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> k_in: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> z_temp_out: array<vec2<f32>>;

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
        z_temp_out[idx] = vec2<f32>(0.0, 0.0);
        return;
    }

    let z_val = z_in[idx];
    let k_val = k_in[idx];

    let multiplier = rk4_config.coeff * rk4_config.dt;

    z_temp_out[idx] = vec2<f32>(
        z_val.x + multiplier * k_val.x,
        z_val.y + multiplier * k_val.y
    );
}
