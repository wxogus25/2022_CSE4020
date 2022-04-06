from curses import KEY_F9
import gc
import glfw
from OpenGL.GL import *
import numpy as np


gComposedM = np.array([[1., .0, .0], [.0, 1., .0], [.0, .0, 1.]])


def press_Q():
    global gComposedM
    gComposedM = np.array([[1., .0, -.1],
                           [.0, 1., .0], [.0, .0, 1.]]) @ gComposedM
    return


def press_E():
    global gComposedM
    gComposedM = np.array([[1., .0, .1],
                           [.0, 1., .0], [.0, .0, 1.]]) @ gComposedM
    return


def press_A():
    global gComposedM
    t = 10. / 360. * np.pi * 2
    gComposedM = gComposedM @ np.array([[np.cos(t), -np.sin(t), 0],
                                        [np.sin(t), np.cos(t), .0], [.0, .0, 1.]])
    return


def press_D():
    global gComposedM
    t = -10. / 360. * np.pi * 2
    gComposedM = gComposedM @ np.array([[np.cos(t), -np.sin(t), 0],
                                        [np.sin(t), np.cos(t), .0], [.0, .0, 1.]])
    return


def press_1():
    global gComposedM
    gComposedM = np.array([[1., .0, .0], [.0, 1., .0], [.0, .0, 1.]])
    return


def press_W():
    global gComposedM
    gComposedM = np.array(
        [[.9, .0, .0], [.0, 1., .0], [.0, .0, 1.]]) @ gComposedM
    return


def press_S():
    global gComposedM
    t = 10. / 360. * np.pi * 2
    gComposedM = np.array([[np.cos(t), -np.sin(t), 0],
                           [np.sin(t), np.cos(t), .0], [.0, .0, 1.]]) @ gComposedM
    return


def key_callback(window, key, scancode, action, mods):
    global gComposedM
    keyMap = {glfw.KEY_Q: press_Q, glfw.KEY_E: press_E, glfw.KEY_A: press_A,
              glfw.KEY_D: press_D, glfw.KEY_1: press_1, glfw.KEY_W: press_W, glfw.KEY_S: press_S}
    if action == glfw.PRESS:
        keyMap[key]()


def render(T):
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()
    # draw cooridnate
    glBegin(GL_LINES)
    glColor3ub(255, 0, 0)
    glVertex2fv(np.array([0., 0.]))
    glVertex2fv(np.array([1., 0.]))
    glColor3ub(0, 255, 0)
    glVertex2fv(np.array([0., 0.]))
    glVertex2fv(np.array([0., 1.]))
    glEnd()
    # draw triangle
    glBegin(GL_TRIANGLES)
    glColor3ub(255, 255, 255)
    glVertex2fv((T @ np.array([.0, .5, 1.]))[:-1])
    glVertex2fv((T @ np.array([.0, .0, 1.]))[:-1])
    glVertex2fv((T @ np.array([.5, .0, 1.]))[:-1])
    glEnd()


def main():
    if not glfw.init():
        return

    window = glfw.create_window(480, 480, "2018009116-3-1", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.set_key_callback(window, key_callback)
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    while not glfw.window_should_close(window):
        glfw.poll_events()
        render(gComposedM)
        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()
