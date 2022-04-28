#!/usr/bin/env python3
# -*- coding: utf-8 -*
# see examples below
# also read all the comments below.

# from asyncio.windows_events import NULL
from email.mime import image
import os
import sys
import pdb  # use pdb.set_trace() for debugging
# or use code.interact(local=dict(globals(), **locals()))  for debugging.
import code
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image


def normalize(vec):
    scala = np.sqrt(sum(list(map(lambda x: np.square(x), vec.tolist()))))
    return np.array(list(map(lambda x: np.divide(x, scala), vec.tolist()))).astype(np.float64)


class Color:
    def __init__(self, R, G, B):
        self.color = np.array([R, G, B]).astype(np.float64)

    # Gamma corrects this color.
    # @param gamma the gamma value to use (2.2 is generally used).
    def gammaCorrect(self, gamma):
        inverseGamma = 1.0 / gamma
        self.color = np.power(self.color, inverseGamma)

    def toUINT8(self):
        return (np.clip(self.color, 0, 1)*255).astype(np.uint8)


class Shader:
    def __init__(self, name, type, dC):
        self.diffuseColor = dC.copy()
        self.type = type
        self.name = name

    def set_Phong(self, sC, E):
        self.specularColor = sC.copy()
        self.exponent = E


class Surface:
    def __init__(self, type, ref):
        self.type = type
        self.ref = ref

    def set_sphere(self, radius, center):
        self.radius = radius
        self.center = center.copy()

    def set_box(self, minPt, maxPt):
        self.minPt = minPt.copy()
        self.maxPt = maxPt.copy()

    def get_normal(self, point):
        if self.type == 'Sphere':
            return normalize(point - self.center)
        else:
            temp = [.0, .0, .0]
            for i in range(3):
                if np.isclose(point[i], self.minPt[i]) or np.isclose(point[i], self.maxPt[i]):
                    # if np.fabs(point[i] - self.minPt[i]) <= sys.float_info.epsilon or np.fabs(point[i] - self.maxPt[i]) <= sys.float_info.epsilon:
                    temp[i] = 1.
                    return np.array(temp)
            print(point, self.minPt, self.maxPt)


class Light:
    def __init__(self, position, intensity):
        self.position = position.copy()
        self.intensity = intensity.copy()


class Ray:
    def __init__(self, p, d):
        self.p = p.astype(np.float64)
        self.d = normalize(d)

    def point(self, t):
        return self.p.astype(np.float64) + (np.float64(t))*self.d.astype(np.float64)


def slabIntersection(ray, surf, axis):
    Min = np.array([surf.minPt[axis[0]], surf.minPt[axis[1]]]
                   ).astype(np.float64)
    Max = np.array([surf.maxPt[axis[0]], surf.maxPt[axis[1]]]
                   ).astype(np.float64)
    d = ray.d
    p = ray.p
    Tmin = np.array([.0, .0]).astype(np.float64)
    Tmax = np.array([.0, .0]).astype(np.float64)

    for i in range(2):
        if d[axis[i]] == .0:
            if Min[i] <= p[axis[i]] <= Max[i]:
                Tmin[i] = -np.inf
                Tmax[i] = np.inf
            else:
                Tmin[i] = np.inf
                Tmax[i] = -np.inf
        else:
            Tmin[i] = min((Min[i] - p[axis[i]])/d[axis[i]],
                          (Max[i] - p[axis[i]])/d[axis[i]])
            Tmax[i] = max((Min[i] - p[axis[i]])/d[axis[i]],
                          (Max[i] - p[axis[i]])/d[axis[i]])
            # Tmin[i] = max(0, (Min[i] - p[axis[i]])/d[axis[i]])
            # Tmax[i] = max(0, (Min[i] - p[axis[i]])/d[axis[i]])

    return max(Tmin[0], Tmin[1]), min(Tmax[0], Tmax[1])


def collisionToSurface(ray, surf):
    t = None
    if surf.type == 'Sphere':
        p = ray.p - surf.center
        Tm = np.float64(-p @ ray.d)
        innerRoot = np.float64(np.square(p @ ray.d) - p @
                               p + np.square(surf.radius))
        if innerRoot < .0:
            return None
        deltaT = np.sqrt(innerRoot)
        t = Tm + deltaT if Tm - deltaT <= .0 else Tm - deltaT
    else:
        tMin, tMax = slabIntersection(ray, surf, [0, 1])
        for i in range(2):
            l, r = slabIntersection(ray, surf, [i, 2])
            tMin = max(tMin, l)
            tMax = min(tMax, r)

        if tMin > tMax or tMax <= .0:
            return None
        t = tMax if tMin <= .0 else tMin
        # f = open("test.txt", 'w')
        # f.write(ray.point(t))
        # f.close()
    return t


def collisionToSurfaces(ray, surfaces):
    tBest = np.inf
    firstSurface = None
    for surf in surfaces:
        t = collisionToSurface(ray, surf)
        if t is not None and t < tBest and t > 0:
            tBest = t
            firstSurface = surf

    return firstSurface, tBest


def isShadow(light, point, surfaces):
    d = normalize(light.position - point)
    p = point
    ray = Ray(p, d)
    surf, _ = collisionToSurfaces(ray, surfaces)
    return surf is not None


def makeColor(ray, surfaces, lights, shaders):
    surf, t = collisionToSurfaces(ray, surfaces)
    if surf is None:
        return Color(0, 0, 0)
    I = [0, 0, 0]
    shader = shaders[surf.ref]
    vec_n = surf.get_normal(ray.point(t)).astype(np.float64)
    vec_v = -ray.d

    for light in lights:
        vec_l = normalize(light.position - ray.point(t))
        intensity = shader.diffuseColor * \
            light.intensity * max(.0, vec_n @ vec_l)
        if shader.type == 'Phong':
            vec_h = normalize(vec_v + vec_l)
            intensity += shader.specularColor * light.intensity * \
                np.power(max(0, vec_n @ vec_h), shader.exponent)

        if not isShadow(light, ray.point(t) - ray.d * 0.000001, surfaces):
            I += intensity

    return Color(I[0], I[1], I[2])


def main():
    tree = ET.parse(sys.argv[1])
    root = tree.getroot()

    # values setting
    # set default values
    viewDir = np.array([0, 0, -1]).astype(np.float)
    viewUp = np.array([0, 1, 0]).astype(np.float)
    # you can safely assume this. (no examples will use shifted perspective camera)
    projNormal = -1*viewDir
    viewWidth = 1.0
    viewHeight = 1.0
    projDistance = 1.0
    # how bright the light is.
    intensity = np.array([1, 1, 1]).astype(np.float)

    # set image value
    imgSize = np.array(root.findtext('image').split()).astype(np.int64)

    # set camera values
    camera = root.find('camera')
    viewPoint = np.array(camera.findtext(
        'viewPoint').split()).astype(np.float64)
    viewDir = normalize(np.array(camera.findtext(
        'viewDir').split()).astype(np.float64))
    projNormal = normalize(np.array(camera.findtext(
        'projNormal').split()).astype(np.float64))
    viewUp = normalize(np.array(camera.findtext(
        'viewUp').split()).astype(np.float64))
    projDistance = np.float64(camera.findtext('projDistance'))
    viewWidth = np.float64(camera.findtext('viewWidth'))
    viewHeight = np.float64(camera.findtext('viewHeight'))

    # set shaders values
    shaders = {}
    for c in root.findall('shader'):
        name = c.get('name')
        type = c.get('type')
        diffuseColor = np.array(c.findtext(
            'diffuseColor').split()).astype(np.float64)
        shad = Shader(name, type, diffuseColor)

        if type == 'Phong':
            specularColor = np.array(c.findtext(
                'specularColor').split()).astype(np.float64)
            exponent = np.float64(c.findtext('exponent'))
            shad.set_Phong(specularColor, exponent)

        shaders.update({name: shad})

    # set surfaces values
    surfaces = []
    for c in root.findall('surface'):
        type = c.get('type')
        ref = c.find('shader').get('ref')
        surf = Surface(type, ref)

        if type == 'Sphere':
            radius = np.float64(c.findtext('radius'))
            center = np.array(c.findtext('center').split()).astype(np.float64)
            surf.set_sphere(radius, center)
        else:
            minPt = np.array(c.findtext('minPt').split()).astype(np.float64)
            maxPt = np.array(c.findtext('maxPt').split()).astype(np.float64)
            surf.set_box(minPt, maxPt)

        surfaces.append(surf)

    # set lights values
    lights = []
    for c in root.findall('light'):
        position = np.array(c.findtext('position').split()).astype(np.float64)
        intensity = np.array(c.findtext(
            'intensity').split()).astype(np.float64)
        light = Light(position, intensity)
        lights.append(light)

    # Create an empty image
    channels = 3
    img = np.zeros((imgSize[1], imgSize[0], channels), dtype=np.uint8)
    img[:, :] = 0

    # Ray Tracing
    # view setting
    vec_u = normalize(np.cross(viewDir, viewUp))
    vec_v = normalize(np.cross(vec_u, viewDir))
    vec_w = -viewDir

    # s = viewPoint - projDistance * vec_w
    # p = s
    # d = normalize(s - viewPoint)
    # ray = Ray(p, d)
    # makeColor(ray, surfaces, lights, shaders)

    for y in np.arange(imgSize[1]):
        for x in np.arange(imgSize[0]):
            nx = (x + 0.5) / imgSize[0]
            ny = (y + 0.5) / imgSize[1]
            u = -viewWidth/2 + viewWidth * nx
            v = -viewHeight/2 + viewHeight * ny

            s = viewPoint + u * vec_u + v * vec_v - projDistance * vec_w
            p = viewPoint
            d = normalize(s - viewPoint)
            ray = Ray(p, d)
            color = makeColor(ray, surfaces, lights, shaders)
            color.gammaCorrect(2.2)
            img[imgSize[1] - 1 - y][x] = color.toUINT8()

    rawimg = Image.fromarray(img, 'RGB')
    # rawimg.save('out.png')
    rawimg.save(sys.argv[1]+'.png')


if __name__ == "__main__":
    main()
