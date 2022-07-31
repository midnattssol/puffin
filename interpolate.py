#!/usr/bin/env python3.10
"""Easing functions taken from easings.net.

Included easings
=======
ease_back
ease_bounce
ease_circ
ease_cubic
ease_elastic
ease_expo
ease_linear
ease_quad
ease_quart
ease_quint
ease_sine
"""

import math
import typing as t

# ===| Constants |===

_C1 = 1.70158
_C2 = _C1 * 1.525
_C3 = _C1 + 1
_C4 = (2 * math.pi) / 3
_C5 = (2 * math.pi) / 4.5

# ===| Functions |===


def clamp(x, min_, max_):
    return max(min_, min(max_, x))


def lerp(a, b, t):
    return t * (b - a) + a


def oscillate(n, pause=0.5):
    """Transforms an increasing value into a value that smoothly oscillates between 0 and 1."""

    period = 2 * pause + 2
    n %= period

    if n < pause:
        return 0
    if n < pause + 1:
        return n - pause
    return 1 - oscillate(n - period / 2, pause)


def easing_average(easings, weights=None):
    """Returns a unary function which returns the average of the unary functions."""

    def easing_inner(x: float) -> float:
        weights = list(
            weights if weights is not None else (map(((lambda _=None: 1)), easings))
        )

        return sum(f(x) * w for f, w in zip(easings, weights, strict=True)) / (
            sum(weights)
        )

    return easing_inner


def _bounce_out(x: float) -> float:
    x = max(min(x, 1), 0)
    n1 = 7.5625
    d1 = 2.75

    if x < 1 / d1:
        return n1 * x * x
    elif x < 2 / d1:
        return n1 * (x - 1.5 / d1) * x + 0.75
    elif x < 2.5 / d1:
        return n1 * (x - 2.25 / d1) * x + 0.9375
    else:
        return n1 * (x - 2.625 / d1) * x + 0.984375


def linear(x: float) -> float:
    x = max(min(x, 1), 0)
    return x


def in_quad(x: float) -> float:
    x = max(min(x, 1), 0)
    return x * x


def out_quad(x: float) -> float:
    x = max(min(x, 1), 0)
    return 1 - (1 - x) * (1 - x)


def in_out_quad(x: float) -> float:
    x = max(min(x, 1), 0)
    return 2 * x * x if x < 0.5 else 1 - pow(-2 * x + 2, 2) / 2


def in_cubic(x: float) -> float:
    x = max(min(x, 1), 0)
    return x**3


def out_cubic(x: float) -> float:
    x = max(min(x, 1), 0)
    return 1 - pow(1 - x, 3)


def in_out_cubic(x: float) -> float:
    x = max(min(x, 1), 0)
    return 4 * x**3 if x < 0.5 else 1 - pow(-2 * x + 2, 3) / 2


def in_quart(x: float) -> float:
    x = max(min(x, 1), 0)
    return x**4


def out_quart(x: float) -> float:
    x = max(min(x, 1), 0)
    return 1 - pow(1 - x, 4)


def in_out_quart(x: float) -> float:
    x = max(min(x, 1), 0)
    return 8 * x**4 if x < 0.5 else 1 - pow(-2 * x + 2, 4) / 2


def in_quint(x: float) -> float:
    x = max(min(x, 1), 0)
    return x**5


def out_quint(x: float) -> float:
    x = max(min(x, 1), 0)
    return 1 - pow(1 - x, 5)


def in_out_quint(x: float) -> float:
    x = max(min(x, 1), 0)
    return 16 * x**5 if x < 0.5 else 1 - pow(-2 * x + 2, 5) / 2


def in_sine(x: float) -> float:
    x = max(min(x, 1), 0)
    if x == 0:
        return 0
    if x == 1:
        return 1
    return 1 - math.cos((x * math.pi) / 2)


def out_sine(x: float) -> float:
    x = max(min(x, 1), 0)
    if x == 0:
        return 0
    if x == 1:
        return 1
    return math.sin((x * math.pi) / 2)


def in_out_sine(x: float) -> float:
    x = max(min(x, 1), 0)
    if x == 0:
        return 0
    if x == 1:
        return 1
    return -(math.cos(math.pi * x) - 1) / 2


def in_expo(x: float) -> float:
    x = max(min(x, 1), 0)
    return 0 if x == 0 else pow(2, 10 * x - 10)


def out_expo(x: float) -> float:
    x = max(min(x, 1), 0)
    return 1 if x == 1 else 1 - pow(2, -10 * x)


def in_out_expo(x: float) -> float:
    x = max(min(x, 1), 0)
    if x == 0:
        return 0
    if x == 1:
        return 1
    if x < 0.5:
        return pow(2, 20 * x - 10) / 2
    return (2 - pow(2, -20 * x + 10)) / 2


def in_circ(x: float) -> float:
    x = max(min(x, 1), 0)
    if x == 0:
        return 0
    if x == 1:
        return 1
    return 1 - math.sqrt(1 - pow(x, 2))


def out_circ(x: float) -> float:
    x = max(min(x, 1), 0)
    return math.sqrt(1 - pow(x - 1, 2))


def in_out_circ(x: float) -> float:
    x = max(min(x, 1), 0)
    if x == 0:
        return 0
    if x == 1:
        return 1
    return (
        (1 - math.sqrt(1 - pow(2 * x, 2))) / 2
        if x < 0.5
        else (math.sqrt(1 - pow(-2 * x + 2, 2)) + 1) / 2
    )


def in_back(x: float) -> float:
    x = max(min(x, 1), 0)
    if x == 0:
        return 0
    if x == 1:
        return 1
    return _C3 * x**3 - _C1 * x**2


def out_back(x: float) -> float:
    x = max(min(x, 1), 0)
    if x == 0:
        return 0
    if x == 1:
        return 1
    return 1 + _C3 * pow(x - 1, 3) + _C1 * pow(x - 1, 2)


def in_out_back(x: float) -> float:
    x = max(min(x, 1), 0)
    if x == 0:
        return 0
    if x == 1:
        return 1
    return (
        (pow(2 * x, 2) * ((_C2 + 1) * 2 * x - _C2)) / 2
        if x < 0.5
        else (pow(2 * x - 2, 2) * ((_C2 + 1) * (x * 2 - 2) + _C2) + 2) / 2
    )


def in_elastic(x: float) -> float:
    x = max(min(x, 1), 0)
    if x == 0:
        return 0
    if x == 1:
        return 1
    return -pow(2, 10 * x - 10) * math.sin((x * 10 - 10.75) * _C4)


def out_elastic(x: float) -> float:
    x = max(min(x, 1), 0)
    if x == 0:
        return 0
    if x == 1:
        return 1
    return pow(2, -10 * x) * math.sin((x * 10 - 0.75) * _C4) + 1


def in_out_elastic(x: float) -> float:
    x = max(min(x, 1), 0)
    if x == 0:
        return 0
    if x == 1:
        return 1
    if x < 0.5:
        return -(pow(2, 20 * x - 10) * math.sin((20 * x - 11.125) * _C5)) / 2
    return (pow(2, -20 * x + 10) * math.sin((20 * x - 11.125) * _C5)) / 2 + 1


def in_bounce(x: float) -> float:
    x = max(min(x, 1), 0)
    if x == 0:
        return 0
    if x == 1:
        return 1
    return 1 - _bounce_out(1 - x)


def out_bounce(x: float) -> float:
    x = max(min(x, 1), 0)
    if x == 0:
        return 0
    if x == 1:
        return 1
    return (
        (1 - _bounce_out(1 - 2 * x)) / 2
        if x < 0.5
        else (1 + _bounce_out(2 * x - 1)) / 2
    )


def in_out_bounce(x: float) -> float:
    x = max(min(x, 1), 0)
    if x == 0:
        return 0
    if x == 1:
        return 1
    if x < 0.5:
        return (1 - out_bounce(1 - 2 * x)) / 2
    return (1 + out_bounce(2 * x - 1)) / 2


def oscillating(a, b, easing=in_out_back):
    return lambda t: lerp(a, b, easing(oscillate(t)))
