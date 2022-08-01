#!/usr/bin/env python3.10
"""Utilities."""
import dataclasses as dc
import time
import typing as t

import colour
import nerdfonts
import numpy as np
import copy

TAU = np.pi * 2


def get_nf_icon(name):
    """Get a nerd font icon from its name."""

    def nerdify(name):
        return name.removeprefix("nf-").replace("-", "_")

    if name is None or (norm_name := nerdify(name)) not in nerdfonts.icons:
        norm_name = nerdify("fa-question_circle_o")

    return nerdfonts.icons[norm_name]


# ===| Coordinates |===


def polar2cartesian(coords):
    """Convert polar coordinates to cartesian coordinates."""
    c = np.cos(coords[1])
    s = np.sin(coords[1])
    r = coords[0]

    return np.array([c * r, s * r])


def cartesian2polar(coords):
    """Convert cartesian coordinates to polar coordinates."""
    r = np.linalg.norm(coords)
    if r == 0:
        return np.zeros(2)

    phi = np.arctan2(*coords[::-1])
    return np.array([r, phi])


def rotate2d(arr, angle):
    """Rotate a 2D array with respect to (0, 0)."""

    c = np.cos(angle)
    s = np.sin(angle)

    rotated = np.dot(arr, np.array([[c, -s], [s, c]]))

    return rotated


# ===| Interpolation |===


def rotating_interpolation(a, b, t):
    """Interpolates 2 2D points with a rotation."""

    delta = b - a
    angle_image_width = 0.25

    return rotate2d(delta * t, TAU * (1 - t) * angle_image_width) + a


# ===| Colors |===


def normalize_color(color):
    """Normalize a color into RGBA."""
    if color is None:
        return color
    elif isinstance(color, str):
        return np.array([*colour.hex2rgb(color), 1])
    elif len(color) == 3:
        return np.array([*color, 1])
    elif len(color) == 4:
        assert isinstance(color, np.ndarray)
        return color

    raise NotImplementedError()


class Color:
    BLACK = None
    WHITE = None
    ACCENT_0 = None
    ACCENT_1 = None


# ===| Misc |===


def index_or(container, obj):
    return container.index(obj) if obj in container else None


def lerp(a, b, t):
    return (b - a) * t + a
