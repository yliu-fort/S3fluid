import ctypes
from kivy.core.window import Window
from kivy.graphics.opengl import glGetString, GL_RENDERER, GL_VERSION, GL_EXTENSIONS

# We will need the raw OpenGL library to call glReadPixels with GL_FLOAT
# because Kivy's opengl.pyx hardcodes GL_UNSIGNED_BYTE for glReadPixels.
import sys

def get_gl_library():
    try:
        if sys.platform.startswith('win'):
            return ctypes.windll.opengl32
        elif sys.platform.startswith('darwin'):
            from ctypes.util import find_library
            path = find_library('OpenGL')
            if path:
                return ctypes.cdll.LoadLibrary(path)
            return ctypes.cdll.LoadLibrary('/System/Library/Frameworks/OpenGL.framework/OpenGL')
        else:
            from ctypes.util import find_library
            path = find_library('GL')
            if path:
                return ctypes.CDLL(path)
            return ctypes.CDLL('libGL.so.1')
    except Exception as e:
        print(f"Warning: Could not load OpenGL library: {e}")
        return None

gl_lib = get_gl_library()

def check_gl_extensions():
    if not Window:
        raise RuntimeError("Kivy Window must be initialized before checking GL extensions.")

    # In Kivy, glGetString returns a Python bytes object, not a raw C pointer.
    try:
        exts_bytes = glGetString(GL_EXTENSIONS)
        if exts_bytes:
            exts = exts_bytes.decode('utf-8', errors='ignore')
            if "GL_ARB_texture_float" not in exts and "GL_EXT_color_buffer_float" not in exts:
                version_bytes = glGetString(GL_VERSION)
                if version_bytes:
                    ver = version_bytes.decode('utf-8', errors='ignore')
                    if not any(ver.startswith(str(i)) for i in range(3, 10)):
                        print(f"Warning: GL_ARB_texture_float or GL_EXT_color_buffer_float not found. Version: {ver}")
    except Exception as e:
        print(f"Failed to check GL extensions explicitly: {e}")
