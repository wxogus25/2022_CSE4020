from curses import KEY_F9
import glfw
from OpenGL.GL import *
import numpy as np

Primitive_Type = GL_LINE_LOOP


def key_callback(window, key, scancode, action, mods):
    global Primitive_Type
    keymap = {glfw.KEY_0: GL_POLYGON, glfw.KEY_1: GL_POINTS, glfw.KEY_2: GL_LINES, glfw.KEY_3: GL_LINE_STRIP, glfw.KEY_4: GL_LINE_LOOP,
              glfw.KEY_5: GL_TRIANGLES, glfw.KEY_6: GL_TRIANGLE_STRIP, glfw.KEY_7: GL_TRIANGLE_FAN, glfw.KEY_8: GL_QUADS, glfw.KEY_9: GL_QUAD_STRIP}

    Primitive_Type = keymap[key]


def render():
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()
    glBegin(Primitive_Type)
    points = np.linspace(0, 2*np.pi, 12, endpoint=False)
    for point in points:
        glVertex2fv(np.array([np.cos(point), np.sin(point)]))
    glEnd()


def main():
    if not glfw.init():
        return

    window = glfw.create_window(480, 480, "2018009116-2-2", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.set_key_callback(window, key_callback)
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    while not glfw.window_should_close(window):
        glfw.poll_events()
        render()
        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()
