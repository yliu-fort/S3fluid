from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.label import Label
from kivy.uix.togglebutton import ToggleButton
from kivy.clock import Clock

from kivy_gpgpu_pde.renderer import MeshRenderer
from kivy_gpgpu_pde.numpy_sht import NumPySHT
from kivy_gpgpu_pde.gpu_sht import GPUSHT
from kivy_gpgpu_pde.integrator import RK4Integrator
from kivy_gpgpu_pde.physics import NavierStokesRHS
import numpy as np
from kivy_gpgpu_pde.gl_utils import FloatFbo

class S3FluidApp(App):
    def build(self):
        self.root = BoxLayout(orientation='vertical')

        # 1. Main visualization area
        self.renderer = MeshRenderer(size_hint=(1, 0.8))
        self.root.add_widget(self.renderer)

        # 2. Controls UI
        controls = BoxLayout(orientation='horizontal', size_hint=(1, 0.2))

        # Play/Pause
        self.btn_play = ToggleButton(text="Play", state='down')
        self.btn_play.bind(state=self.on_play_pause)
        controls.add_widget(self.btn_play)

        # Resolution Slider
        res_layout = BoxLayout(orientation='vertical')
        self.lbl_res = Label(text="Resolution (l_max): 16")
        self.slider_res = Slider(min=8, max=64, value=16, step=8)
        self.slider_res.bind(value=self.on_res_change)
        res_layout.add_widget(self.lbl_res)
        res_layout.add_widget(self.slider_res)
        controls.add_widget(res_layout)

        # Wireframe Toggle
        self.btn_wireframe = ToggleButton(text="Wireframe")
        self.btn_wireframe.bind(state=self.on_wireframe)
        controls.add_widget(self.btn_wireframe)

        # Info Label (CFL etc)
        self.lbl_info = Label(text="CFL: 0.00")
        controls.add_widget(self.lbl_info)

        self.root.add_widget(controls)

        # Simulation State
        self.l_max = int(self.slider_res.value)
        self.dt = 0.005
        self.time = 0.0

        self._init_simulation()

        # Start clock
        if self.btn_play.state == 'down':
            Clock.schedule_interval(self.update, 0.0)

        return self.root

    def _init_simulation(self):
        self.numpy_sht = NumPySHT(self.l_max)
        self.gpu_sht = GPUSHT(self.numpy_sht)

        # Setup Physics and Integrator
        # To ensure we have a working physics loop for Phase 4 rendering verification,
        # we will use the pure Python NumPy_RHS inside a generic RK4 Integrator since Shader RHS
        # might still be incomplete from Phase 3 or difficult to wire entirely in Kivy
        # without actual FBO ping-pong classes.
        self.physics = NavierStokesRHS(self.numpy_sht, nu=1e-4)

        # We'll step the coefficients themselves for simplicity in Phase 4.
        self.state_fbo = FloatFbo(size=(self.numpy_sht.n_lon, self.numpy_sht.n_lat))

        # Initialize initial vorticity field (e.g., spherical harmonic Y_3^2)
        lats = self.numpy_sht.lats[:, np.newaxis]
        lons = self.numpy_sht.lons[np.newaxis, :]

        # Creating a dynamic initial field
        zeta_grid = np.sin(3 * lats) * np.cos(2 * lons)
        self.current_coeffs = self.numpy_sht.forward_sht(zeta_grid)

        # Upload
        self._update_fbo_from_coeffs()
        self.renderer.build_mesh(self.numpy_sht)

    def _update_fbo_from_coeffs(self):
        grid = self.numpy_sht.inverse_sht(self.current_coeffs)
        rgba_grid = np.zeros((self.numpy_sht.n_lat, self.numpy_sht.n_lon, 4), dtype=np.float32)
        rgba_grid[..., 0] = grid
        self.state_fbo.write_pixels_float(rgba_grid)
        self.renderer.texture = self.state_fbo.texture

    def update(self, dt):
        # Perform 1 pure python RK4 step on the spectral coefficients
        k1 = self.physics.numpy_nonl(self.current_coeffs)
        k2 = self.physics.numpy_nonl(self.current_coeffs + 0.5 * self.dt * k1)
        k3 = self.physics.numpy_nonl(self.current_coeffs + 0.5 * self.dt * k2)
        k4 = self.physics.numpy_nonl(self.current_coeffs + self.dt * k3)

        self.current_coeffs += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.time += self.dt

        self._update_fbo_from_coeffs()

        # Read back max velocity to compute CFL
        # For Phase 4, we approximate max velocity from the current vorticity grid
        try:
            rgba_array = self.state_fbo.read_pixels_float()
            zeta_grid = rgba_array[..., 0]
            # Fast approximation: |u| ~ |zeta| / l_max (dimensionally)
            max_vel_approx = np.max(np.abs(zeta_grid)) / (self.l_max + 1)

            # CFL = u * dt / dx.
            # dx on sphere is roughly pi / n_lat
            dx = np.pi / self.numpy_sht.n_lat
            cfl = (max_vel_approx * self.dt) / dx
        except Exception as e:
            print(f"Failed to calculate CFL: {e}")
            cfl = 0.0

        self.lbl_info.text = f"CFL: {cfl:.3f}\nTime: {self.time:.3f}"

        # Trigger mesh redraw via rotation
        self.renderer.canvas.ask_update()

    def on_play_pause(self, instance, state):
        if state == 'down':
            Clock.schedule_interval(self.update, 0.0)
        else:
            Clock.unschedule(self.update)

    def on_res_change(self, instance, value):
        new_l_max = int(value)
        if new_l_max == self.l_max:
            return

        self.lbl_res.text = f"Resolution (l_max): {new_l_max}"

        # 1. Pause Clock simulation
        was_playing = False
        if self.btn_play.state == 'down':
            Clock.unschedule(self.update)
            was_playing = True

        print(f"Switching resolution from l_max={self.l_max} to l_max={new_l_max}")

        # 2. Download vorticity field zeta using fbo.pixels
        # We need the current grid data as a float array. We use FloatFbo's read_pixels_float
        try:
            rgba_array = self.state_fbo.read_pixels_float()
            zeta_grid = rgba_array[..., 0] # Extract the real physical field

            # 3. Get spectral coefficients zeta_{lm}^{old} with sht_old.forward_sht
            zeta_coeffs_old = self.numpy_sht.forward_sht(zeta_grid)
        except Exception as e:
            print(f"Could not read FBO state: {e}. Reinitializing from scratch.")
            zeta_coeffs_old = None

        # 4. Release old FBOs to prevent VRAM leak
        self.state_fbo.release()

        # 5. Create new NumPySHT with the new l_max
        old_numpy_sht = self.numpy_sht
        self.l_max = new_l_max
        self.numpy_sht = NumPySHT(self.l_max)

        # Recreate GPU SHT arrays/textures
        self.gpu_sht = GPUSHT(self.numpy_sht)

        # 6. Zero-pad or truncate zeta_{lm}^{old} to fit new dimensions
        zeta_coeffs_new = np.zeros((self.numpy_sht.num_m, self.numpy_sht.l_max + 1), dtype=complex)

        if zeta_coeffs_old is not None:
            # We copy over matching (m, l) coefficients.
            for i_m_new, m in enumerate(self.numpy_sht.m_values):
                # Find index of m in the old m_values
                i_m_old_idx = np.where(old_numpy_sht.m_values == m)[0]
                if len(i_m_old_idx) > 0:
                    i_m_old = i_m_old_idx[0]

                    # Copy matching l values up to min(old_l_max, new_l_max)
                    max_l_common = min(old_numpy_sht.l_max, self.numpy_sht.l_max)

                    if m <= max_l_common:
                        # Slice size
                        slice_size = max_l_common - m + 1
                        zeta_coeffs_new[i_m_new, m:m+slice_size] = zeta_coeffs_old[i_m_old, m:m+slice_size]

            # 7. Transform back to grid
            new_zeta_grid = self.numpy_sht.inverse_sht(zeta_coeffs_new)
        else:
            # Initialize some dummy data if readback failed
            lats = self.numpy_sht.lats[:, np.newaxis]
            lons = self.numpy_sht.lons[np.newaxis, :]
            new_zeta_grid = np.sin(3 * lats) * np.cos(2 * lons)

        # Initialize new FBO
        self.state_fbo = FloatFbo(size=(self.numpy_sht.n_lon, self.numpy_sht.n_lat))
        rgba_grid_new = np.zeros((self.numpy_sht.n_lat, self.numpy_sht.n_lon, 4), dtype=np.float32)
        rgba_grid_new[..., 0] = new_zeta_grid
        self.state_fbo.write_pixels_float(rgba_grid_new)

        # 8. Update Mesh and start simulation again
        self.renderer.build_mesh(self.numpy_sht)
        self.renderer.texture = self.state_fbo.texture

        if was_playing:
            Clock.schedule_interval(self.update, 0.0)

    def on_wireframe(self, instance, state):
        self.renderer.wireframe = (state == 'down')

if __name__ == '__main__':
    S3FluidApp().run()
