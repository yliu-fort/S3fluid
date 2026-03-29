// Forward FFT for purely real input grid.
// Maps grid(j, k) -> F(j, m)
//
// Workgroup setup:
// x: Longitude (m or k index)
// y: Latitude (j index)
// z: 1

@group(0) @binding(0) var<storage, read> grid_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> freq_out: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> phi: array<f32>;

struct Config {
    nlat: u32,
    nlon: u32,
    lmax: u32,
    pad: u32
}

@group(1) @binding(0) var<uniform> config: Config;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let m = global_id.x;
    let j = global_id.y;

    if (m > config.lmax || j >= config.nlat) {
        return;
    }

    var sum_re = 0.0;
    var sum_im = 0.0;

    for (var k = 0u; k < config.nlon; k = k + 1u) {
        let angle = -f32(m) * phi[k];
        let val = grid_in[j * config.nlon + k];
        sum_re = sum_re + val * cos(angle);
        sum_im = sum_im + val * sin(angle);
    }

    let out_idx = j * (config.lmax + 1u) + m;
    freq_out[out_idx] = vec2<f32>(sum_re, sum_im);
}
