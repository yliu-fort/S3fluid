import pytest
import numpy as np
import os

# Set environment variable for Kivy to use headless provider in tests
os.environ['KIVY_WINDOW'] = 'sdl2'
os.environ['KIVY_GL_BACKEND'] = 'sdl2'

# Import Kivy after setting environment variables
from kivy.core.window import Window
from kivy.graphics import RenderContext, Color, Rectangle, Fbo
from kivy.graphics.texture import Texture
from kivy_gpgpu_pde.fbo_utils import GPGPUFbo
from kivy_gpgpu_pde.numpy_sht import NumPySHT
from kivy_gpgpu_pde.sht_gpu import SHT_GPU

def test_gpgpu_fbo_precision():
    """
    UT-01: FBO environment & precision check
    Writes high dynamic range values using a simple shader to the FBO,
    then reads it back via our float32 reader to ensure no precision is lost.
    """
    # Create the GPGPU FBO
    size = (4, 4)
    fbo = GPGPUFbo(size=size)

    # Kivy FBO acts like an instruction group.
    # We create a RenderContext to bind a custom shader.
    rc = RenderContext(use_parent_modelview=True, use_parent_projection=True)

    # We need a simple shader to write specific values
    # e.g., output color = (1e5, 1e-8, -50.5, 1.0)
    fs = '''
    $HEADER$
    void main(void) {
        gl_FragColor = vec4(100000.0, 0.00000001, -50.5, 1.0);
    }
    '''
    rc.shader.fs = fs

    with rc:
        Color(1, 1, 1, 1)
        # Draw a full-screen quad to trigger fragment shader for every pixel
        Rectangle(pos=(0, 0), size=size)

    fbo.add(rc)
    fbo.draw()

    # Read the float32 pixels back
    pixels = fbo.read_pixels_float32()

    # Cleanup to avoid memory leaks
    fbo.remove(rc)
    # fbo.release() requires fbo to be bound or it is just destroying. Kivy FBO handles cleanup.

    # Verify the read values match what was written in the shader
    assert pixels.shape == (4, 4, 4)

    # Test top-left pixel
    R, G, B, A = pixels[0, 0]

    assert np.isclose(R, 100000.0, rtol=1e-5), f"Red channel loss of precision: {R}"
    assert np.isclose(G, 1e-8, rtol=1e-5, atol=1e-10), f"Green channel loss of precision: {G}"
    assert np.isclose(B, -50.5, rtol=1e-5), f"Blue channel loss of precision: {B}"
    assert np.isclose(A, 1.0, rtol=1e-5), f"Alpha channel mismatch: {A}"


def test_sht_gpu_upload_matrices():
    """
    UT-02: Precomputed data texturization.
    Ensures that the initialized textures for SHT computation don't crash
    and are valid float32 textures.
    """
    nsht = NumPySHT(l_max=8)
    gpu_sht = SHT_GPU(numpy_sht=nsht)

    assert gpu_sht.tex_legendre is not None
    assert gpu_sht.tex_legendre.colorfmt == 'rgba'
    assert gpu_sht.tex_legendre.bufferfmt == 'float'


def test_sht_gpu_forward():
    """
    UT-03: Forward SHT operator pass initialization.
    Verifies that the forward SHT placeholder outputs an FBO successfully.
    """
    nsht = NumPySHT(l_max=8)
    gpu_sht = SHT_GPU(numpy_sht=nsht)

    # We pass a dummy grid texture
    dummy_grid = Texture.create(size=(nsht.n_lon, nsht.n_lat))
    spec_fbo = gpu_sht.forward_sht(dummy_grid)

    assert spec_fbo is not None
    pixels = spec_fbo.read_pixels_float32()

    # Verify our mock shader gives Red
    R, G, B, A = pixels[0, 0]
    assert np.isclose(R, 1.0)
    assert np.isclose(G, 0.0)

def test_sht_gpu_inverse():
    """
    UT-04: Inverse SHT operator pass initialization.
    Verifies that the inverse SHT placeholder outputs an FBO successfully.
    """
    nsht = NumPySHT(l_max=8)
    gpu_sht = SHT_GPU(numpy_sht=nsht)

    dummy_spec = Texture.create(size=(nsht.l_max + 1, nsht.num_m))
    grid_fbo = gpu_sht.inverse_sht(dummy_spec)

    assert grid_fbo is not None
    pixels = grid_fbo.read_pixels_float32()

    # Verify our mock shader gives Green
    R, G, B, A = pixels[0, 0]
    assert np.isclose(G, 1.0)
    assert np.isclose(R, 0.0)

def test_sht_gpu_inverse_grad():
    """
    UT-06: Gradient operator pass initialization.
    Verifies that the inverse SHT gradient placeholder outputs an FBO successfully.
    """
    nsht = NumPySHT(l_max=8)
    gpu_sht = SHT_GPU(numpy_sht=nsht)

    dummy_spec = Texture.create(size=(nsht.l_max + 1, nsht.num_m))
    grid_fbo = gpu_sht.inverse_sht_grad(dummy_spec)

    assert grid_fbo is not None
    pixels = grid_fbo.read_pixels_float32()

    # Verify our mock shader gives Blue
    R, G, B, A = pixels[0, 0]
    assert np.isclose(B, 1.0)
    assert np.isclose(R, 0.0)
