import os
import pytest
os.environ['KIVY_NO_ARGS'] = '1'
import numpy as np
os.environ['KIVY_WINDOW'] = 'sdl2'
os.environ['KIVY_GL_BACKEND'] = 'sdl2'

from kivy.core.window import Window
from src.texture_utils import ndarray_to_texture
from src.fbo_utils import GL_FLOAT, GL_RGBA, GL_TEXTURE_2D
from src.gl_utils import gl_lib
import ctypes

def test_precomputed_data_ut02():
    # UT-02: Precomputed Data Textureization
    # Load CPU computed Legendre matrices/weights into Texture and read back to assert correctness.

    np.random.seed(42)
    width, height = 4, 4

    # Generate random data representing e.g., spherical harmonics weights
    # Range of typical Legendre polynomials can be varied
    mock_weights = np.random.uniform(-10.0, 10.0, size=(height, width, 2)).astype(np.float32)

    # Convert to texture
    tex = ndarray_to_texture(mock_weights)

    assert tex.size == (width, height)
    assert tex.colorfmt == 'rgba'
    assert tex.bufferfmt == 'float'

    # Read back the texture directly using glGetTexImage since Kivy doesn't provide a direct read from Texture
    # Kivy FBO `pixels` uses glReadPixels, but here we can just use PyOpenGL or ctypes to read the texture data
    if gl_lib is None:
        pytest.skip("gl_lib not available, cannot read back float texture")

    tex.bind()

    # void glGetTexImage(GLenum target, GLint level, GLenum format, GLenum type, void *pixels);
    read_data = np.zeros((height, width, 4), dtype=np.float32)
    gl_lib.glGetTexImage(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        GL_FLOAT,
        read_data.ctypes.data_as(ctypes.c_void_p)
    )

    # Assert RG channels exactly match
    assert np.allclose(read_data[..., 0], mock_weights[..., 0], rtol=1e-6, atol=1e-8)
    assert np.allclose(read_data[..., 1], mock_weights[..., 1], rtol=1e-6, atol=1e-8)

    # The padding channels B and A should be 0.0 and 1.0
    assert np.allclose(read_data[..., 2], 0.0, atol=1e-8)
    assert np.allclose(read_data[..., 3], 1.0, atol=1e-8)

if __name__ == '__main__':
    test_precomputed_data_ut02()
    print("UT-02 passed!")
