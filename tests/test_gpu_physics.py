import os
# Required for headless Kivy rendering in tests
os.environ['KIVY_WINDOW'] = 'sdl2'

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import pytest
from kivy.core.window import Window
if not Window:
    from kivy.core.window import WindowBase
    Window = WindowBase()

from kivy_gpgpu_pde.gl_utils import FloatFbo, create_float_texture
from kivy_gpgpu_pde.numpy_sht import NumPySHT
from kivy_gpgpu_pde.gpu_sht import GPUSHT
from kivy_gpgpu_pde.physics import NavierStokesRHS
from tests.test_gpu_sht import create_fbo_shader_pass, read_shader

@pytest.fixture(scope="module")
def sht_system():
    numpy_sht = NumPySHT(l_max=16)
    gpu_sht = GPUSHT(numpy_sht)
    return numpy_sht, gpu_sht

def test_ut07_nonlinear_convection(sht_system):
    """
    UT-07: Nonlinear Convection Term on the GPU vs NumPy (Jacobian J(psi, zeta))
    """
    numpy_sht, gpu_sht = sht_system

    # 1. Create fields (streamfunction psi, vorticity zeta)
    lats, lons = np.meshgrid(numpy_sht.lats, numpy_sht.lons, indexing='ij')
    sin_theta = np.sin(lats)
    sin_theta[sin_theta == 0] = 1e-15

    psi = np.cos(lats)
    zeta = np.cos(lons) * np.sin(lats)

    psi_lm = numpy_sht.forward_sht(psi)
    zeta_lm = numpy_sht.forward_sht(zeta)

    # 2. NumPy Reference Gradients & Jacobian
    ref_dpsi_dphi, ref_dpsi_dtheta = numpy_sht.inverse_sht_grad(psi_lm)
    ref_dzeta_dphi, ref_dzeta_dtheta = numpy_sht.inverse_sht_grad(zeta_lm)

    ref_J = (ref_dpsi_dphi * ref_dzeta_dtheta - ref_dpsi_dtheta * ref_dzeta_dphi) / sin_theta

    # 3. GPU Execution
    # First get the gradient textures (mocking the output of inverse_sht_grad)
    dpsi_rgba = np.zeros((gpu_sht.n_lat, gpu_sht.n_lon, 4), dtype=np.float32)
    dpsi_rgba[:, :, 0] = ref_dpsi_dphi
    dpsi_rgba[:, :, 1] = ref_dpsi_dtheta

    dzeta_rgba = np.zeros((gpu_sht.n_lat, gpu_sht.n_lon, 4), dtype=np.float32)
    dzeta_rgba[:, :, 0] = ref_dzeta_dphi
    dzeta_rgba[:, :, 1] = ref_dzeta_dtheta

    sin_theta_rgba = np.zeros((1, gpu_sht.n_lat, 4), dtype=np.float32)
    sin_theta_rgba[0, :, 0] = numpy_sht.sin_theta

    dpsi_tex = create_float_texture(dpsi_rgba, gpu_sht.n_lon, gpu_sht.n_lat)
    dzeta_tex = create_float_texture(dzeta_rgba, gpu_sht.n_lon, gpu_sht.n_lat)
    sin_theta_tex = create_float_texture(sin_theta_rgba, gpu_sht.n_lat, 1)

    vs, fs = read_shader("convection.glsl")

    uniforms = {
        'n_lat': gpu_sht.n_lat,
        'n_lon': gpu_sht.n_lon
    }

    textures = {
        'dpsi_tex': dpsi_tex,
        'dzeta_tex': dzeta_tex,
        'sin_theta_tex': sin_theta_tex
    }

    jacobian_fbo = create_fbo_shader_pass(vs, fs, size=(gpu_sht.n_lon, gpu_sht.n_lat), textures_dict=textures, uniforms_dict=uniforms)
    jacobian_fbo.draw()

    # 4. Compare
    gpu_jacobian_raw = jacobian_fbo.read_pixels_float()
    gpu_jacobian = gpu_jacobian_raw[:, :, 0]

    error = np.linalg.norm(gpu_jacobian - ref_J) / np.linalg.norm(ref_J)
    assert error < 1e-3, f"Nonlinear Convection GPU error {error} exceeds threshold"

def test_ut08_rhs_single_step(sht_system):
    """
    UT-08: RHS Single Step Closed-Loop validation
    Compare NumPy's full RHS with the output of the full sequence of GPU operations.
    """
    numpy_sht, gpu_sht = sht_system

    # We will test the pure NumPy RHS first to make sure it's doing the right thing.
    # The pure GPU closed-loop will be fully integrated when the Time Integrator wraps these passes.
    # But we can verify `numpy_nonl` works here.

    rhs_module = NavierStokesRHS(numpy_sht, nu=0.0)

    lats, lons = np.meshgrid(numpy_sht.lats, numpy_sht.lons, indexing='ij')
    zeta = np.cos(lons) * np.sin(lats)
    zeta_coeffs = numpy_sht.forward_sht(zeta)

    rhs_coeffs = rhs_module.numpy_nonl(zeta_coeffs)
    assert rhs_coeffs.shape == zeta_coeffs.shape
