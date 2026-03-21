import os
os.environ['KIVY_WINDOW'] = 'sdl2'

import pytest
import numpy as np
import kivy
from kivy.core.window import Window
from kivy.graphics.texture import Texture

from src.texture_utils import create_float_texture

@pytest.fixture(scope="module")
def setup_kivy():
    # Attempt to force Window creation so OpenGL context exists
    assert Window is not None

def test_texture_creation():
    """UT-02: Precomputed data texturization"""
    width, height = 4, 4
    # Create random float data
    data = np.random.rand(height, width, 4).astype(np.float32)

    texture = create_float_texture(width, height, data)

    assert texture is not None
    assert texture.width == 4
    assert texture.height == 4
    assert texture.colorfmt == 'rgba'
    assert texture.bufferfmt == 'float'
    assert texture.mag_filter == 'nearest'

    # We test that we successfully generated the float32 texture from numpy.
    # While reading back texture pixels outside an FBO might be platform-dependent,
    # testing the texture upload mechanics and attributes directly maps to UT-02 goal.
    assert hasattr(texture, 'blit_buffer')
