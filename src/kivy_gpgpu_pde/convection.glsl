---VERTEX SHADER---
#version 330 core

in vec2 vPosition;
in vec2 vTexCoords0;

out vec2 tex_coord;

void main(void) {
    tex_coord = vTexCoords0;
    gl_Position = vec4(vPosition, 0.0, 1.0);
}

---FRAGMENT SHADER---
#version 330 core

in vec2 tex_coord;
out vec4 frag_color;

uniform int n_lon;
uniform int n_lat;
uniform sampler2D dpsi_tex;  // R: dpsi/dphi, G: dpsi/dtheta
uniform sampler2D dzeta_tex; // R: dzeta/dphi, G: dzeta/dtheta
uniform sampler2D sin_theta_tex; // R: sin(theta)

void main(void) {
    int lon_idx = int(floor(gl_FragCoord.x));
    int lat_idx = int(floor(gl_FragCoord.y));

    if (lon_idx >= n_lon || lat_idx >= n_lat) {
        frag_color = vec4(0.0);
        return;
    }

    vec2 grad_psi = texelFetch(dpsi_tex, ivec2(lon_idx, lat_idx), 0).rg;
    vec2 grad_zeta = texelFetch(dzeta_tex, ivec2(lon_idx, lat_idx), 0).rg;

    float sin_t = texelFetch(sin_theta_tex, ivec2(lat_idx, 0), 0).r;

    // Protect against division by zero (poles)
    if (sin_t < 1e-15) {
        sin_t = 1e-15;
    }

    // J(psi, zeta) = (dpsi/dphi * dzeta/dtheta - dpsi/dtheta * dzeta/dphi) / sin(theta)
    float j_val = (grad_psi.x * grad_zeta.y - grad_psi.y * grad_zeta.x) / sin_t;

    frag_color = vec4(j_val, 0.0, 0.0, 1.0);
}
