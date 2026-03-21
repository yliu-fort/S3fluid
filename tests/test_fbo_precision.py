import os
import pytest
os.environ['KIVY_NO_ARGS'] = '1'
import numpy as np
os.environ['KIVY_WINDOW'] = 'sdl2'
os.environ['KIVY_GL_BACKEND'] = 'sdl2'

from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, ClearColor, ClearBuffers, RenderContext
from src.fbo_utils import create_float_fbo, fbo_to_ndarray

FS_UT01 = """
$HEADER$
void main(void) {
    gl_FragColor = vec4(1.23e5, 1.45e-8, -1.0, 1.0);
}
"""

def test_fbo_precision_ut01():
    fbo = create_float_fbo(2, 2)

    rc = RenderContext(use_parent_projection=True, use_parent_modelview=True)
    rc.shader.fs = FS_UT01

    with fbo:
        ClearColor(0.0, 0.0, 0.0, 0.0)
        ClearBuffers()

    fbo.add(rc)
    rc.add(Rectangle(pos=(0, 0), size=(2, 2)))

    fbo.draw()
    fbo.remove(rc)

    data = fbo_to_ndarray(fbo)

    assert data.shape == (2, 2, 4)
    assert data.dtype == np.float32

    # Check values with relative error < 1e-6
    assert np.allclose(data[0,0,0], 1.23e5, rtol=1e-6, atol=0.0)
    assert np.allclose(data[0,0,1], 1.45e-8, rtol=1e-6, atol=1e-10)
    assert np.allclose(data[0,0,2], -1.0, rtol=1e-6, atol=0.0)
    assert np.allclose(data[0,0,3], 1.0, rtol=1e-6, atol=0.0)

if __name__ == '__main__':
    test_fbo_precision_ut01()
    print("UT-01 passed!")
