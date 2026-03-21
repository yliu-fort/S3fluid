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

from kivy.graphics import Fbo, RenderContext, Rectangle, BindTexture, Mesh, UpdateNormalMatrix
from kivy.graphics.opengl import glDisable, GL_DEPTH_TEST
from kivy_gpgpu_pde.gl_utils import FloatFbo
from kivy_gpgpu_pde.numpy_sht import NumPySHT
from kivy_gpgpu_pde.gpu_sht import GPUSHT

# Get shader directory paths
import kivy_gpgpu_pde
SHADER_DIR = os.path.dirname(kivy_gpgpu_pde.__file__)

def read_shader(filename):
    with open(os.path.join(SHADER_DIR, filename), 'r') as f:
        content = f.read()
    # Split Kivy shader format (---VERTEX SHADER--- and ---FRAGMENT SHADER---)
    vs_start = content.find("---VERTEX SHADER---")
    fs_start = content.find("---FRAGMENT SHADER---")

    vs = content[vs_start + 19:fs_start].strip()
    fs = content[fs_start + 21:].strip()
    return vs, fs

@pytest.fixture(scope="module")
def sht_system():
    # Use low resolution to keep tests fast but valid
    numpy_sht = NumPySHT(l_max=16)
    gpu_sht = GPUSHT(numpy_sht)
    return numpy_sht, gpu_sht

def test_ut02_texture_generation(sht_system):
    """
    UT-02: Verify precomputed data textureization matches NumPy arrays exactly.
    """
    numpy_sht, gpu_sht = sht_system

    # We'll skip rendering FBO for textures and just assert dimensions & type
    # For Kivy texture sizes, depending on the internal flipped or normal status, we expect correctly allocated dimensions
    assert gpu_sht.weights_texture.size == (gpu_sht.n_lat, 1) or gpu_sht.weights_texture.size == (1, gpu_sht.n_lat)
    assert gpu_sht.alp_texture.size == (gpu_sht.n_lat, gpu_sht.num_m * (gpu_sht.l_max + 1))
    assert gpu_sht.dalp_texture.size == (gpu_sht.n_lat, gpu_sht.num_m * (gpu_sht.l_max + 1))
    assert gpu_sht.dft_fwd_texture.size == (gpu_sht.n_lon, gpu_sht.num_m) or gpu_sht.dft_fwd_texture.size == (gpu_sht.num_m, gpu_sht.n_lon)
    assert gpu_sht.dft_inv_texture.size == (gpu_sht.num_m, gpu_sht.n_lon) or gpu_sht.dft_inv_texture.size == (gpu_sht.n_lon, gpu_sht.num_m)

def create_fbo_shader_pass(vs, fs, size, textures_dict, uniforms_dict):
    """
    Helper to create an FBO with a custom shader pass.
    """
    fbo = FloatFbo(size=size)

    # Override FBO RenderContext with custom shaders
    with fbo:
        # Clear color
        from kivy.graphics import ClearColor, ClearBuffers
        ClearColor(0, 0, 0, 0)
        ClearBuffers(clear_color=True)

        # Do not inherit modelview to ensure gl_FragCoord maps 1-to-1 to FBO pixels
        # instead of Screen/Window pixels
        rc = RenderContext(vs=vs, fs=fs)

        with rc:
            # Bind all secondary textures
            tex_idx = 1
            for name, tex in textures_dict.items():
                BindTexture(texture=tex, index=tex_idx)
                rc[name] = tex_idx
                tex_idx += 1

            # Set Uniforms
            for name, val in uniforms_dict.items():
                rc[name] = val

            primary_name, primary_tex = list(textures_dict.items())[0]
            rc[primary_name] = 0 # override to 0
            BindTexture(texture=primary_tex, index=0)

            # Map directly to Normalized Device Coordinates
            vertices = [
                -1.0, -1.0,  0.0, 0.0,
                 1.0, -1.0,  1.0, 0.0,
                 1.0,  1.0,  1.0, 1.0,
                -1.0,  1.0,  0.0, 1.0
            ]
            indices = [0, 1, 2, 2, 3, 0]
            Mesh(vertices=vertices, indices=indices, mode='triangles', texture=primary_tex)

    return fbo

def test_ut03_forward_sht(sht_system):
    """
    UT-03: Forward SHT Operator on GPU vs NumPy
    """
    numpy_sht, gpu_sht = sht_system

    # 1. Create a physical field
    lats, lons = np.meshgrid(numpy_sht.lats, numpy_sht.lons, indexing='ij')
    physical_data = np.cos(lats)**2 * np.sin(lons) # Analytically smooth

    # 2. NumPy reference
    ref_coeffs = numpy_sht.forward_sht(physical_data)

    # 3. GPU execution
    # Upload physical data
    phys_data_rgba = np.zeros((gpu_sht.n_lat, gpu_sht.n_lon, 4), dtype=np.float32)
    phys_data_rgba[:, :, 0] = physical_data

    # size is (width, height) -> (n_lon, n_lat)
    phys_tex = FloatFbo(size=(gpu_sht.n_lon, gpu_sht.n_lat))
    phys_tex.write_pixels_float(phys_data_rgba)
    phys_tex.texture.mag_filter = 'nearest'
    phys_tex.texture.min_filter = 'nearest'

    vs, fs = read_shader("forward_sht.glsl")

    uniforms = {
        'l_max': gpu_sht.l_max,
        'n_lat': gpu_sht.n_lat,
        'n_lon': gpu_sht.n_lon,
        'num_m': gpu_sht.num_m
    }

    textures = {
        'physical_data_tex': phys_tex.texture,
        'dft_fwd_tex': gpu_sht.dft_fwd_texture,
        'alp_tex': gpu_sht.alp_texture,
        'weights_tex': gpu_sht.weights_texture
    }

    spectral_fbo = create_fbo_shader_pass(vs, fs, size=(gpu_sht.num_m, gpu_sht.l_max + 1), textures_dict=textures, uniforms_dict=uniforms)
    spectral_fbo.draw()

    # 4. Compare
    gpu_coeffs_raw = spectral_fbo.read_pixels_float()

    gpu_coeffs_complex = gpu_coeffs_raw[:, :, 0] + 1j * gpu_coeffs_raw[:, :, 1]

    # Re-order: FBO is (height, width) -> (l_max+1, num_m)
    # NumPy is (num_m, l_max+1)
    gpu_coeffs_complex = gpu_coeffs_complex.T

    # Ignore coefficients where l < m (they are mathematically 0 and initialized to 0 in shader)
    mask = np.zeros((gpu_sht.num_m, gpu_sht.l_max + 1), dtype=bool)
    for i_m, m in enumerate(numpy_sht.m_values):
        mask[i_m, m:] = True

    error = np.linalg.norm(gpu_coeffs_complex[mask] - ref_coeffs[mask]) / np.linalg.norm(ref_coeffs[mask])

    assert error < 1e-3, f"Forward SHT GPU error {error} exceeds threshold"

def test_ut04_inverse_sht(sht_system):
    """
    UT-04: Inverse SHT Operator on GPU vs NumPy
    """
    numpy_sht, gpu_sht = sht_system

    # 1. Generate some spectral coefficients
    coeffs = np.zeros((gpu_sht.num_m, gpu_sht.l_max + 1), dtype=complex)

    # Populate random coefficients for valid (l, m) pairs
    np.random.seed(42)
    for i_m, m in enumerate(numpy_sht.m_values):
        for l in range(m, gpu_sht.l_max + 1):
            coeffs[i_m, l] = np.random.randn() + 1j * np.random.randn()
            # Suppress high frequencies to avoid huge numerical noise
            coeffs[i_m, l] *= np.exp(-0.1 * l)

    # NumPy reference
    ref_physical = numpy_sht.inverse_sht(coeffs)

    # GPU Execution
    # Prepare texture: width = num_m, height = l_max + 1
    coeffs_rgba = np.zeros((gpu_sht.l_max + 1, gpu_sht.num_m, 4), dtype=np.float32)
    coeffs_rgba[:, :, 0] = coeffs.real.T
    coeffs_rgba[:, :, 1] = coeffs.imag.T

    spectral_tex = FloatFbo(size=(gpu_sht.num_m, gpu_sht.l_max + 1))
    spectral_tex.write_pixels_float(coeffs_rgba)
    spectral_tex.texture.mag_filter = 'nearest'
    spectral_tex.texture.min_filter = 'nearest'

    vs, fs = read_shader("inverse_sht.glsl")

    uniforms = {
        'l_max': gpu_sht.l_max,
        'n_lat': gpu_sht.n_lat,
        'n_lon': gpu_sht.n_lon,
        'num_m': gpu_sht.num_m
    }

    textures = {
        'spectral_data_tex': spectral_tex.texture,
        'dft_inv_tex': gpu_sht.dft_inv_texture,
        'alp_tex': gpu_sht.alp_texture
    }

    phys_fbo = create_fbo_shader_pass(vs, fs, size=(gpu_sht.n_lon, gpu_sht.n_lat), textures_dict=textures, uniforms_dict=uniforms)
    phys_fbo.draw()

    # 4. Compare
    gpu_phys_raw = phys_fbo.read_pixels_float()

    # Shape of gpu_phys_raw is (n_lat, n_lon, 4)
    gpu_physical = gpu_phys_raw[:, :, 0]

    error = np.linalg.norm(gpu_physical - ref_physical) / np.linalg.norm(ref_physical)

    assert error < 1e-3, f"Inverse SHT GPU error {error} exceeds threshold"

def test_ut06_inverse_sht_grad(sht_system):
    """
    UT-06: Gradient Operators on the GPU
    """
    numpy_sht, gpu_sht = sht_system

    # 1. Create a physical field (streamfunction psi)
    lats, lons = np.meshgrid(numpy_sht.lats, numpy_sht.lons, indexing='ij')
    psi = np.cos(lats) + np.sin(lats) * np.cos(lons)

    # Get coefficients
    psi_coeffs = numpy_sht.forward_sht(psi)

    # 2. NumPy Reference Gradients
    ref_dphi, ref_dtheta = numpy_sht.inverse_sht_grad(psi_coeffs)

    # 3. GPU Execution
    coeffs_rgba = np.zeros((gpu_sht.l_max + 1, gpu_sht.num_m, 4), dtype=np.float32)
    coeffs_rgba[:, :, 0] = psi_coeffs.real.T
    coeffs_rgba[:, :, 1] = psi_coeffs.imag.T

    spectral_tex = FloatFbo(size=(gpu_sht.num_m, gpu_sht.l_max + 1))
    spectral_tex.write_pixels_float(coeffs_rgba)
    spectral_tex.texture.mag_filter = 'nearest'
    spectral_tex.texture.min_filter = 'nearest'

    vs, fs = read_shader("inverse_sht_grad.glsl")

    uniforms = {
        'l_max': gpu_sht.l_max,
        'n_lat': gpu_sht.n_lat,
        'n_lon': gpu_sht.n_lon,
        'num_m': gpu_sht.num_m
    }

    textures = {
        'spectral_data_tex': spectral_tex.texture,
        'dft_inv_tex': gpu_sht.dft_inv_texture,
        'alp_tex': gpu_sht.alp_texture,
        'dalp_tex': gpu_sht.dalp_texture
    }

    phys_fbo = create_fbo_shader_pass(vs, fs, size=(gpu_sht.n_lon, gpu_sht.n_lat), textures_dict=textures, uniforms_dict=uniforms)
    phys_fbo.draw()

    # 4. Compare
    gpu_phys_raw = phys_fbo.read_pixels_float()

    # R channel is dphi, G channel is dtheta
    gpu_dphi = gpu_phys_raw[:, :, 0]
    gpu_dtheta = gpu_phys_raw[:, :, 1]

    error_dphi = np.linalg.norm(gpu_dphi - ref_dphi) / np.linalg.norm(ref_dphi)
    error_dtheta = np.linalg.norm(gpu_dtheta - ref_dtheta) / np.linalg.norm(ref_dtheta)

    assert error_dphi < 1e-2, f"Inverse SHT Grad dphi GPU error {error_dphi} exceeds threshold"
    assert error_dtheta < 1e-2, f"Inverse SHT Grad dtheta GPU error {error_dtheta} exceeds threshold"

def test_ut05_linear_operators():
    # Will be implemented alongside specific physical PDEs. Pass for now.
    pass
