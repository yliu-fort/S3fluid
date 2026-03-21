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

// Input coordinates
in vec2 tex_coord;

// Output float texture
out vec4 frag_color;

// System inputs
uniform int l_max;
uniform int n_lat;
uniform int n_lon;
uniform int num_m;

// Texture Inputs
uniform sampler2D spectral_data_tex; // The spectral coefficients (num_m x (l_max+1)). R: real, G: imag
uniform sampler2D dft_inv_tex;       // The Inverse DFT Matrix (num_m x n_lon). R: scale*cos, G: scale*sin
uniform sampler2D alp_tex;           // The Associated Legendre Polynomials
uniform sampler2D dalp_tex;          // The Derivative of Associated Legendre Polynomials

void main(void) {
    // Current texel coordinate specifies which physical point (lon_idx, lat_idx) we are calculating
    // In our physical texture, x is the lon index, y is the lat index.
    int lon_idx = int(floor(gl_FragCoord.x));
    int lat_idx = int(floor(gl_FragCoord.y));

    if (lon_idx >= n_lon || lat_idx >= n_lat) {
        frag_color = vec4(0.0);
        return;
    }

    float f_val_dphi = 0.0;
    float f_val_dtheta = 0.0;

    for (int m_idx = 0; m_idx < num_m; m_idx++) {
        int m = m_idx;

        vec2 f_m = vec2(0.0);       // \sum_{l=m}^{l_max} C_{l,m} P_l^m
        vec2 df_dtheta_m = vec2(0.0); // \sum_{l=m}^{l_max} C_{l,m} dP_l^m/d\theta

        // Sum over l
        for (int l_idx = m; l_idx <= l_max; l_idx++) {
            // Get spectral coefficient
            vec2 c_lm = texelFetch(spectral_data_tex, ivec2(m_idx, l_idx), 0).rg;

            // Get ALP and DALP values
            int y_alp_int = m_idx * (l_max + 1) + l_idx;

            float p_lm = texelFetch(alp_tex, ivec2(lat_idx, y_alp_int), 0).r;
            float dp_lm = texelFetch(dalp_tex, ivec2(lat_idx, y_alp_int), 0).r;

            f_m += c_lm * p_lm;
            df_dtheta_m += c_lm * dp_lm;
        }

        // Multiply by inverse DFT basis e^{i m \phi_k}
        vec2 basis = texelFetch(dft_inv_tex, ivec2(m_idx, lon_idx), 0).rg;

        // For dphi, we multiply f_m by (i * m) before inverse DFT
        // f_m * (i * m) = (f_m.x + i f_m.y) * (i * m) = -m * f_m.y + i * m * f_m.x
        vec2 df_dphi_m = vec2(-float(m) * f_m.y, float(m) * f_m.x);

        float real_part_dphi = df_dphi_m.x * basis.x - df_dphi_m.y * basis.y;
        float real_part_dtheta = df_dtheta_m.x * basis.x - df_dtheta_m.y * basis.y;

        f_val_dphi += real_part_dphi;
        f_val_dtheta += real_part_dtheta;
    }

    // Output dphi to R channel, dtheta to G channel
    frag_color = vec4(f_val_dphi, f_val_dtheta, 0.0, 1.0);
}
