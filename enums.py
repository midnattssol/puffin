#!/usr/bin/env python3.10
"""Enumerations."""
import enum

import numpy as np


class Direction(enum.Enum):
    VERTICAL = enum.auto()
    HORIZONTAL = enum.auto()


class Packing(enum.Enum):
    START = enum.auto()
    CENTER = enum.auto()
    END = enum.auto()
    NONE = enum.auto()


class Anchor(enum.Flag):
    """There is a better name for this but I forgot.

    It's called a Anchor for now since you could imagine a anchor being stuck into the object at this point."""

    TOP = enum.auto()
    BOTTOM = enum.auto()
    LEFT = enum.auto()
    RIGHT = enum.auto()

    def to_arr(self) -> np.ndarray:
        assert not (self.BOTTOM & self.TOP)
        assert not (self.LEFT & self.RIGHT)

        arr = np.array([0.5, 0.5])

        if self & Anchor.RIGHT:
            arr[0] = 1
        if self & Anchor.LEFT:
            arr[0] = 0
        if self & Anchor.BOTTOM:
            arr[1] = 1
        if self & Anchor.TOP:
            arr[1] = 0

        return arr
