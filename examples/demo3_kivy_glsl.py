import math
import numpy as np
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import RenderContext, Fbo, Rectangle, Color, BindTexture, ClearColor, ClearBuffers
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.clock import Clock

import demo3

# Configuration
LMAX = 31
M = LMAX + 1
L = LMAX + 1
J = LMAX + 1
NLON = 2 * (LMAX + 1)
NU = 1.0e-7
DT = 1.0e-2

# ----------------------------------------------------
# Shaders
# ----------------------------------------------------

# Pass 1 & 2: Frequency sums (L-sum)
# We need to compute: 
# u_theta_freq, u_phi_freq (Pass 1)
# dzeta_theta_freq, dzeta_phi_freq (Pass 2)
# Since we can't easily do multiple render targets in standard Kivy/GLES2 without extensions,
# replacing multi-targets with separate passes, or packing into one FBO if possible.
# Actually, complex values need 2 floats (RG). 
# We can pack 2 complex values into RGBA:
# Freq Vel FBO (RGBA): u_theta_freq.r, u_theta_freq.i, u_phi_freq.r, u_phi_freq.i
# Freq Vort FBO (RGBA): dzeta_theta_freq.r, ... 
# 
# Wait, let's output both in one shader if we render to 2 separate FBOs? No, we need 2 different FBOs, 
# so 2 slightly different shaders, or one shader that branches, but branching is bad. 
# Let's write separate shader strings.

VS_PASSTHROUGH = '''
$HEADER$
void main() {
    gl_Position = projection_mat * modelview_mat * vec4(pos, 1.0);
    tex_coord0 = tex_coords0;
}
'''

FS_FREQ_VEL = '''
$HEADER$
uniform sampler2D tex_zeta; // M x L complex
uniform sampler2D tex_P_all; // (M*L) x J (P, dP, Pw)
uniform sampler2D tex_lap; // M x L (lap, inv_lap)
uniform sampler2D tex_sin_theta; // 1 x J

uniform float M;
uniform float L;

void main() {
    // We are rendering to M x J texture. 
    // Wait, Kivy standard gl_FragCoord is in pixels [0.5, W-0.5].
    // Texture coordinates tex_coord0 are [0, 1].
    
    int m = int(tex_coord0.x * M);
    int j = int(tex_coord0.y * M); // Note: J is same size as M.

    // Get sin_theta
    float sin_t = texture2D(tex_sin_theta, vec2((float(j)+0.5)/M, 0.5)).r;

    vec2 u_theta_freq = vec2(0.0);
    vec2 u_phi_freq = vec2(0.0);

    for(int l=0; l<32; l++) {  // LMAX+1 = 32
        if (l >= m) {
            // Fetch zeta_lm
            vec2 zeta = texture2D(tex_zeta, vec2((float(m)+0.5)/M, (float(l)+0.5)/L)).rg;
            
            // Fetch inv_lap
            float inv_lap = texture2D(tex_lap, vec2((float(m)+0.5)/M, (float(l)+0.5)/L)).g;
            vec2 psi = zeta * inv_lap;

            // Fetch P and dP
            float x_idx = float(m * 32 + l);
            vec4 p_data = texture2D(tex_P_all, vec2((x_idx+0.5)/(M*L), (float(j)+0.5)/M));
            float P_val = p_data.r;
            float dP_val = p_data.g;

            // dpsi_dphi = psi * i * m
            vec2 dpsi_dphi = vec2(-psi.y * float(m), psi.x * float(m));

            // u_theta_freq sum accumulation: P * dpsi_dphi / sin_t
            u_theta_freq += P_val * dpsi_dphi / sin_t;

            // u_phi_freq sum accumulation: -dP * psi
            u_phi_freq += -dP_val * psi;
        }
    }

    gl_FragColor = vec4(u_theta_freq.x, u_theta_freq.y, u_phi_freq.x, u_phi_freq.y);
}
'''

FS_FREQ_VORT = '''
$HEADER$
uniform sampler2D tex_zeta; // M x L complex
uniform sampler2D tex_P_all; // (M*L) x J (P, dP, Pw)
uniform sampler2D tex_sin_theta; // 1 x J

uniform float M;
uniform float L;

void main() {
    int m = int(tex_coord0.x * M);
    int j = int(tex_coord0.y * M); 
    float sin_t = texture2D(tex_sin_theta, vec2((float(j)+0.5)/M, 0.5)).r;

    vec2 dzeta_theta_freq = vec2(0.0);
    vec2 dzeta_phi_freq = vec2(0.0);

    for(int l=0; l<32; l++) {  
        if (l >= m) {
            vec2 zeta = texture2D(tex_zeta, vec2((float(m)+0.5)/M, (float(l)+0.5)/L)).rg;
            
            float x_idx = float(m * 32 + l);
            vec4 p_data = texture2D(tex_P_all, vec2((x_idx+0.5)/(M*L), (float(j)+0.5)/M));
            float P_val = p_data.r;
            float dP_val = p_data.g;

            vec2 dzeta_dphi = vec2(-zeta.y * float(m), zeta.x * float(m));

            dzeta_theta_freq += dP_val * zeta;
            dzeta_phi_freq += P_val * dzeta_dphi;
        }
    }

    gl_FragColor = vec4(dzeta_theta_freq.x, dzeta_theta_freq.y, dzeta_phi_freq.x, dzeta_phi_freq.y);
}
'''

# Pass 3: Spatial Advection (M-sum)
# Render to NLON x J (R channel)
FS_ADV_SPATIAL = '''
$HEADER$
uniform sampler2D tex_freq_vel; // M x J (RGBA)
uniform sampler2D tex_freq_vort; // M x J (RGBA)
uniform sampler2D tex_sin_theta;
uniform float NLON;
uniform float M;

const float PI = 3.1415926535897932384626433832795;

void main() {
    int k = int(tex_coord0.x * NLON);
    int j = int(tex_coord0.y * M); // J=M
    float sin_t = texture2D(tex_sin_theta, vec2((float(j)+0.5)/M, 0.5)).r;

    float phi_k = 2.0 * PI * float(k) / NLON;

    float u_theta = 0.0;
    float u_phi = 0.0;
    float dzeta_theta = 0.0;
    float dzeta_phi = 0.0;

    for(int m=0; m<32; m++) {  
        vec4 vel = texture2D(tex_freq_vel, vec2((float(m)+0.5)/M, tex_coord0.y));
        vec4 vort = texture2D(tex_freq_vort, vec2((float(m)+0.5)/M, tex_coord0.y));
        
        float cos_m_phi = cos(float(m) * phi_k);
        float sin_m_phi = sin(float(m) * phi_k);

        float factor = (m == 0) ? 1.0 : 2.0;

        // freq maps to real val: factor * (real*cos - imag*sin)
        u_theta += factor * (vel.r * cos_m_phi - vel.g * sin_m_phi);
        u_phi   += factor * (vel.b * cos_m_phi - vel.a * sin_m_phi);
        dzeta_theta += factor * (vort.r * cos_m_phi - vort.g * sin_m_phi);
        dzeta_phi   += factor * (vort.b * cos_m_phi - vort.a * sin_m_phi);
    }

    float adv = u_theta * dzeta_theta + u_phi * (dzeta_phi / sin_t);
    gl_FragColor = vec4(adv, 0.0, 0.0, 1.0);
}
'''

# Pass 4: Advection Freq (K-sum)
# Render to M x J (RG)
FS_ADV_FREQ = '''
$HEADER$
uniform sampler2D tex_adv_spatial; // NLON x J
uniform float NLON;
uniform float M;

const float PI = 3.1415926535897932384626433832795;

void main() {
    int m = int(tex_coord0.x * M);
    int j = int(tex_coord0.y * M); 
    
    vec2 adv_freq = vec2(0.0);

    for(int k=0; k<64; k++) { // NLON=64
        float adv = texture2D(tex_adv_spatial, vec2((float(k)+0.5)/NLON, tex_coord0.y)).r;
        float phi_k = 2.0 * PI * float(k) / NLON;
        
        // rfft sum
        float cos_m_phi = cos(-float(m) * phi_k);
        float sin_m_phi = sin(-float(m) * phi_k);

        adv_freq += adv * vec2(cos_m_phi, sin_m_phi);
    }

    gl_FragColor = vec4(adv_freq.x, adv_freq.y, 0.0, 1.0);
}
'''

# Pass 5: Euler Step (L-sum)
# Render to M x L
FS_STEP = '''
$HEADER$
uniform sampler2D tex_adv_freq; // M x J
uniform sampler2D tex_zeta_old; // M x L
uniform sampler2D tex_P_all; // P, dP, Pw
uniform sampler2D tex_lap; // lap, inv_lap
uniform sampler2D tex_filter; // M x L

uniform float M;
uniform float L;
uniform float NLON;
uniform float NU;
uniform float DT;

const float PI = 3.1415926535897932384626433832795;

void main() {
    int m = int(tex_coord0.x * M);
    int l = int(tex_coord0.y * L);
    
    if (l < m) {
        gl_FragColor = vec4(0.0);
        return;
    }

    vec2 adv_lm = vec2(0.0);

    // Sum over j
    for(int j=0; j<32; j++) { // J=32
        vec2 adv_freq = texture2D(tex_adv_freq, vec2(tex_coord0.x, (float(j)+0.5)/M)).rg;
        
        float x_idx = float(m * 32 + l);
        vec4 p_data = texture2D(tex_P_all, vec2((x_idx+0.5)/(M*L), (float(j)+0.5)/M));
        float Pw_val = p_data.b; // Pw

        adv_lm += Pw_val * adv_freq;
    }
    adv_lm *= (2.0 * PI / NLON);

    vec2 zeta_old = texture2D(tex_zeta_old, tex_coord0.xy).rg;
    float lap = texture2D(tex_lap, tex_coord0.xy).r;
    float spec_filter = texture2D(tex_filter, tex_coord0.xy).r;

    vec2 diff = zeta_old * lap * NU;
    vec2 rhs = -adv_lm + diff;

    vec2 zeta_new = (zeta_old + DT * rhs) * spec_filter;
    if (m == 0 && l == 0) {
        zeta_new = vec2(0.0); // mean zero
    }
    
    gl_FragColor = vec4(zeta_new.x, zeta_new.y, 0.0, 1.0);
}
'''

# Visualization Pass: Synthesize Zeta to Spatial field and color map
FS_VIZ = '''
$HEADER$
// This is to be drawn to the screen
uniform sampler2D tex_zeta;
uniform sampler2D tex_P_all;
uniform float M;
uniform float L;

void main() {
    // tex_coord0 maps longitude to X [0, NLON], latitude to Y [0, J]
    float PI = 3.14159265;
    
    // We do full synthesis: M-sum and L-sum right here because we only do it once per frame!
    int k = int(tex_coord0.x * 64.0);
    int j = int(tex_coord0.y * 32.0);
    float phi_k = 2.0 * PI * float(k) / 64.0;

    float zeta_val = 0.0;

    for (int m=0; m<32; m++) {
        vec2 freq = vec2(0.0);
        for (int l=0; l<32; l++) {
            if (l >= m) {
                vec2 zeta_lm = texture2D(tex_zeta, vec2((float(m)+0.5)/M, (float(l)+0.5)/L)).rg;
                float x_idx = float(m * 32 + l);
                float P_val = texture2D(tex_P_all, vec2((x_idx+0.5)/(M*L), (float(j)+0.5)/M)).r;
                freq += P_val * zeta_lm;
            }
        }
        
        float cos_m_phi = cos(float(m) * phi_k);
        float sin_m_phi = sin(float(m) * phi_k);
        float factor = (m == 0) ? 1.0 : 2.0;

        zeta_val += factor * (freq.x * cos_m_phi - freq.y * sin_m_phi);
    }
    
    // RdBu mapping
    float norm_zeta = (zeta_val + 10.0) / 20.0; // scale guessed for visualization
    norm_zeta = clamp(norm_zeta, 0.0, 1.0);
    
    // Simple RdBu approximation
    vec3 color;
    if (norm_zeta < 0.5) {
        color = mix(vec3(0.0, 0.0, 1.0), vec3(1.0, 1.0, 1.0), norm_zeta * 2.0);
    } else {
        color = mix(vec3(1.0, 1.0, 1.0), vec3(1.0, 0.0, 0.0), (norm_zeta - 0.5) * 2.0);
    }

    gl_FragColor = vec4(color, 1.0);
}
'''

def create_float_texture(w, h, data_array, channels=4):
    fmt = 'rgba' if channels == 4 else 'luminance' if channels == 1 else 'rgba'
    if channels == 2:
        fmt = 'rgba'
        padded = np.zeros((h, w, 4), dtype=np.float32)
        padded[..., :2] = data_array
        data_array = padded
        
    tex = Texture.create(size=(w, h), colorfmt=fmt, bufferfmt='float')
    tex.mag_filter = 'nearest'
    tex.min_filter = 'nearest'
    tex.blit_buffer(data_array.astype(np.float32).tobytes(), colorfmt=fmt, bufferfmt='float')
    return tex

class ShaderWidget(Widget):
    def __init__(self, **kwargs):
        self.canvas = RenderContext(fs=FS_VIZ, vs=VS_PASSTHROUGH)
        super().__init__(**kwargs)
        
        self.sht = demo3.SphericalHarmonicTransform(lmax=LMAX)
        self.model = demo3.SphereTurbulence2D(sht=self.sht, nu=NU)
        
        P = self.sht.P
        dP = self.sht.dP
        Pw = self.sht.Pw
        
        p_all_data = np.zeros((J, M * L, 4), dtype=np.float32)
        for j in range(J):
            p_all_data[j, :, 0] = P[j, :, :].reshape(-1)
            p_all_data[j, :, 1] = dP[j, :, :].reshape(-1)
            p_all_data[j, :, 2] = Pw[j, :, :].reshape(-1)
        self.tex_P_all = create_float_texture(M * L, J, p_all_data, channels=4)
        
        lap_data = np.zeros((L, M, 4), dtype=np.float32) 
        lap_data[:, :, 0] = self.sht.lap.T
        lap_data[:, :, 1] = self.sht.inv_lap.T
        self.tex_lap = create_float_texture(M, L, lap_data, channels=4)
        
        sin_t_data = np.zeros((1, M, 4), dtype=np.float32) 
        sin_t_data[0, :, 0] = self.sht.sin_theta
        self.tex_sin_theta = create_float_texture(M, 1, sin_t_data, channels=4)
        
        filter_data = np.zeros((L, M, 4), dtype=np.float32)
        filter_2d = np.broadcast_to(self.model.spec_filter[None, :], (M, L))
        filter_data[:, :, 0] = filter_2d.T
        self.tex_filter = create_float_texture(M, L, filter_data, channels=4)

        zeta_init = self.model.random_initial_vorticity(seed=42, amplitude=1.0)
        zeta_init_data = np.zeros((L, M, 4), dtype=np.float32)
        zeta_init_data[:, :, 0] = zeta_init.real.T
        zeta_init_data[:, :, 1] = zeta_init.imag.T
        
        self.zeta_tex_0 = create_float_texture(M, L, zeta_init_data, channels=4)
        self.zeta_tex_1 = create_float_texture(M, L, np.zeros_like(zeta_init_data), channels=4)
        self.current_zeta_tex = self.zeta_tex_0
        self.next_zeta_tex = self.zeta_tex_1
        
        self.fbo_vel = Fbo(size=(M, J), fs=FS_FREQ_VEL, vs=VS_PASSTHROUGH)
        self.fbo_vort = Fbo(size=(M, J), fs=FS_FREQ_VORT, vs=VS_PASSTHROUGH)
        self.fbo_adv_spatial = Fbo(size=(NLON, J), fs=FS_ADV_SPATIAL, vs=VS_PASSTHROUGH)
        self.fbo_adv_freq = Fbo(size=(M, J), fs=FS_ADV_FREQ, vs=VS_PASSTHROUGH)
        self.fbo_step = Fbo(size=(M, L), fs=FS_STEP, vs=VS_PASSTHROUGH)
        
        # Disable blending on FBOs
        for f in [self.fbo_vel, self.fbo_vort, self.fbo_adv_spatial, self.fbo_adv_freq, self.fbo_step]:
            f.bind()
            f.clear_buffer() # Kivy standard is to just clear context
            f.release()
            
        with self.fbo_vel:
            self.vel_bind_zeta = BindTexture(texture=self.current_zeta_tex, index=1)
            BindTexture(texture=self.tex_P_all, index=2)
            BindTexture(texture=self.tex_lap, index=3)
            BindTexture(texture=self.tex_sin_theta, index=4)
            Rectangle(size=self.fbo_vel.size)
        
        with self.fbo_vort:
            self.vort_bind_zeta = BindTexture(texture=self.current_zeta_tex, index=1)
            BindTexture(texture=self.tex_P_all, index=2)
            BindTexture(texture=self.tex_sin_theta, index=4)
            Rectangle(size=self.fbo_vort.size)
            
        with self.fbo_adv_spatial:
            BindTexture(texture=self.fbo_vel.texture, index=1)
            BindTexture(texture=self.fbo_vort.texture, index=2)
            BindTexture(texture=self.tex_sin_theta, index=4)
            Rectangle(size=self.fbo_adv_spatial.size)
            
        with self.fbo_adv_freq:
            BindTexture(texture=self.fbo_adv_spatial.texture, index=1)
            Rectangle(size=self.fbo_adv_freq.size)
            
        with self.fbo_step:
            BindTexture(texture=self.fbo_adv_freq.texture, index=1)
            self.step_bind_zeta = BindTexture(texture=self.current_zeta_tex, index=2)
            BindTexture(texture=self.tex_P_all, index=3)
            BindTexture(texture=self.tex_lap, index=4)
            BindTexture(texture=self.tex_filter, index=5)
            Rectangle(size=self.fbo_step.size)
            
        with self.canvas:
            self.viz_bind_zeta = BindTexture(texture=self.current_zeta_tex, index=1)
            BindTexture(texture=self.tex_P_all, index=2)
            self.viz_rect = Rectangle(size=self.size)

        self.fbo_vel['tex_zeta'] = 1
        self.fbo_vel['tex_P_all'] = 2
        self.fbo_vel['tex_lap'] = 3
        self.fbo_vel['tex_sin_theta'] = 4
        self.fbo_vel['M'] = float(M)
        self.fbo_vel['L'] = float(L)

        self.fbo_vort['tex_zeta'] = 1
        self.fbo_vort['tex_P_all'] = 2
        self.fbo_vort['tex_sin_theta'] = 4
        self.fbo_vort['M'] = float(M)
        self.fbo_vort['L'] = float(L)

        self.fbo_adv_spatial['tex_freq_vel'] = 1
        self.fbo_adv_spatial['tex_freq_vort'] = 2
        self.fbo_adv_spatial['tex_sin_theta'] = 4
        self.fbo_adv_spatial['NLON'] = float(NLON)
        self.fbo_adv_spatial['M'] = float(M)

        self.fbo_adv_freq['tex_adv_spatial'] = 1
        self.fbo_adv_freq['NLON'] = float(NLON)
        self.fbo_adv_freq['M'] = float(M)

        self.fbo_step['tex_adv_freq'] = 1
        self.fbo_step['tex_zeta_old'] = 2
        self.fbo_step['tex_P_all'] = 3
        self.fbo_step['tex_lap'] = 4
        self.fbo_step['tex_filter'] = 5
        self.fbo_step['M'] = float(M)
        self.fbo_step['L'] = float(L)
        self.fbo_step['NLON'] = float(NLON)
        self.fbo_step['NU'] = float(NU)
        self.fbo_step['DT'] = float(DT)
        
        Window.bind(size=self.on_window_size)
        self.on_window_size(None, Window.size)

    def on_window_size(self, instance, value):
        self.viz_rect.size = value
        self.viz_rect.pos = self.pos

    def update(self, dt):
        self.vel_bind_zeta.texture = self.current_zeta_tex
        self.vort_bind_zeta.texture = self.current_zeta_tex
        self.step_bind_zeta.texture = self.current_zeta_tex
        
        self.fbo_vel.ask_update()
        self.fbo_vel.draw()
        
        self.fbo_vort.ask_update()
        self.fbo_vort.draw()
        
        self.fbo_adv_spatial.ask_update()
        self.fbo_adv_spatial.draw()
        
        self.fbo_adv_freq.ask_update()
        self.fbo_adv_freq.draw()
        
        self.fbo_step.ask_update()
        self.fbo_step.draw()
        
        self.current_zeta_tex = self.fbo_step.texture
        self.viz_bind_zeta.texture = self.current_zeta_tex
        
        self.canvas['M'] = float(M)
        self.canvas['L'] = float(L)
        self.canvas['tex_zeta'] = 1
        self.canvas['tex_P_all'] = 2

class DemoApp(App):
    def build(self):
        widget = ShaderWidget()
        Clock.schedule_interval(widget.update, 1.0 / 60.0)
        return widget

if __name__ == '__main__':
    DemoApp().run()
