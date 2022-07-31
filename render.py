#!/usr/bin/env python3.10
"""An updated render system."""
import dataclasses as dc
import enum
import itertools as it
import time
import typing as t
import math
import cairo
import more_itertools as mit
import numpy as np
import utils
from enums import *
from size import *

TAU = np.pi * 2

# ===| Base classes |===


@dc.dataclass
class Renderable:
    """An object with a parent, anchor, position, and a render method.

    The `__getattribute__` dunder method is overridden in this object, so that any
    attribute which is an instance of `Size` is unwrapped before being returned. This
    is done so that `Size`s can be mixed in with other objects.

    To get the unwrapped attribute, use the `get` method with `unwrap` set to False.
    """

    anchor: Anchor = Anchor(0)
    position: Size = dc.field(default_factory=lambda: Absolute(np.array([0.0, 0.0])))
    parent: t.Any = None

    def render(self, ctx):
        raise NotImplementedError()

    def get(self, attr, unwrap=True):
        gotten = object.__getattribute__(self, attr)
        if not unwrap:
            return gotten

        if not isinstance(gotten, Size):
            return gotten

        if isinstance(gotten, (Relative, Percentage, ParentFunction)):
            parent = object.__getattribute__(self, "parent")
            if parent is None or not hasattr(parent, attr):
                raise ValueError(
                    f"Attribute '{attr}' ({gotten}) of {self} requires a parent with the same attribute, but {parent} lacks it."
                )

            return gotten.get(parent.get(attr), parent)

        return gotten.get()

    def __getattribute__(self, name):
        if (name.startswith("__") and name.endswith("__")) or name == "get":
            return object.__getattribute__(self, name)

        return self.get(name)


@dc.dataclass
class Sized(Renderable):
    """A renderable object with some notion of a size."""

    size: Size = dc.field(default_factory=lambda: Relative(np.array([0.5, 0.5])))

    def get_top_left(self):
        return self.position - self.size * self.anchor.to_arr()


@dc.dataclass
class Container(Sized):
    """A container for organizing other renderable objects."""

    direction: Direction = Direction.HORIZONTAL
    packing: Packing = Packing.NONE
    children: t.List[Renderable] = dc.field(default_factory=list)
    spacing: int = 5

    def make_grid(self, items, scale=[1, 1]):
        assert items
        assert len(set(map(len, items))) == 1

        rows, cols = len(items), len(items[0])

        for i, row in enumerate(items):
            for j, item in enumerate(row):
                item.parent = self
                k = np.array([i / (rows - 1), j / (cols - 1)])
                k -= 0.5
                k *= scale

                item.position = ParentFunction(
                    lambda pos, parent, k=k: k * parent.size + pos
                )

            self.children += row

    def render(self, ctx):
        if self.packing == Packing.NONE or not self.children:
            for child in self.children:
                child.render(ctx)
            return None

        # Recalculate the positions based on the packing.
        i = int(self.direction == Direction.HORIZONTAL)
        i_0 = int(not i)

        rel_pos = np.array([0.0, 0.0])
        rel_pos[i] += self.spacing

        locations = []

        for child in self:
            anchor_array = child.anchor.to_arr()
            anchor_array[i] = 0
            offset = anchor_array * self.size

            # If the child is anchored to the side, it has to be moved closer
            # to the center, since otherwise the child would be centered on the border.
            n = anchor_array[i_0] * 2 - 1
            offset[i_0] -= n * (self.spacing + child.size[i_0] / 2)

            # Get the total of the difference relative to the top right.
            rel_pos[i] += self.spacing
            rel_tot = rel_pos + offset

            # Update the position and render the child.
            locations.append([child, rel_tot])
            rel_pos[i] += child.size[i] + self.spacing

        # Move the anchor to the correct side.
        new_anchor = (
            {
                Packing.START: Anchor.TOP,
                Packing.CENTER: Anchor(0),
                Packing.END: Anchor.BOTTOM,
            }[self.packing]
            if self.direction == Direction.HORIZONTAL
            else {
                Packing.START: Anchor.LEFT,
                Packing.CENTER: Anchor(0),
                Packing.END: Anchor.RIGHT,
            }[self.packing]
        )

        # Center pack by getting the empty space at the bottom
        # and then redistributing it on either side.
        if self.packing == Packing.CENTER:
            p_first = [l for x, l in locations if x == self.children[0]][0]
            p_last = [l for x, l in locations if x == self.children[-1]][0]

            width = p_last + self.children[-1].size - p_first
            avg_pos = (self.size - width) / 2
            avg_pos[i_0] = 0
            locations = [[c, l + avg_pos] for c, l in locations]

        if self.packing == Packing.END:
            size = self.size.copy()
            locations = [[c, size - l] for c, l in locations]

        for child, loc in locations:
            child.position = loc + self.get_top_left()
            child = dc.replace(child, anchor=new_anchor)
            child.render(ctx)

    def __iter__(self):
        return iter(self.children)

    def __len__(self):
        return len(self.children)


@dc.dataclass
class Shape:
    color: ... = dc.field(default_factory=lambda: np.array([0.1, 0.1, 0.1]))
    filled: bool = True
    width: int = 3


@dc.dataclass
class Rectangle(Shape, Container):
    def render(self, ctx):
        ctx.new_path()
        ctx.set_source_rgba(*self.color)
        ctx.rectangle(*self.get_top_left(), *self.get("size"))
        ctx.fill()
        ctx.stroke()

        Container.render(self, ctx)


# DEBUG: Debug this.
@dc.dataclass
class Arc(Renderable, Shape):
    radius: float = 20
    begin_arc: float = 0
    end_arc: float = TAU

    @property
    def size(self):
        return np.array([self.radius, self.radius])

    def render(self, ctx):
        ctx.new_path()
        ctx.set_source_rgba(*self.color)
        center_arr = self.anchor.to_arr() - [0.5, 0.5]

        ctx.arc(
            *(self.position - self.size * center_arr),
            self.radius,
            self.begin_arc,
            self.end_arc,
        )

        if abs(self.begin_arc - self.end_arc) != TAU:
            ctx.line_to(*center)

        ctx.fill()
        ctx.stroke()


# DEBUG: Debug this.
@dc.dataclass
class Text(Renderable):
    color: ... = dc.field(default_factory=lambda: np.array([0.1, 0.1, 0.1]))
    text: str = "Hello world!"
    font_size: int = 20
    font_face: str = "Iosevka Nerd Font"
    font_slant: int = cairo.FONT_SLANT_NORMAL
    font_weight: int = cairo.FONT_WEIGHT_NORMAL

    def render(self, ctx):
        ctx.select_font_face(self.font_face, self.font_slant, self.font_weight)
        ctx.set_font_size(self.font_size)
        ctx.set_source_rgba(*self.color)

        (x, y, width, height, dx, dy) = ctx.text_extents(self.text)
        height = self.font_size

        ctx.move_to(*self.position - ([width, height] * self.anchor.to_arr()))
        ctx.show_text(self.text)

        # DEBUG: Uncomment this to see circles.
        # ctx.set_source_rgba(1, 0.2, 0, 1)
        # ctx.arc(*self.position - ([width, height] * self.anchor.to_arr()), 4, 0, TAU)
        # ctx.fill()
        # ctx.stroke()


class RoundedRectangle(Rectangle):
    radius: float = 8

    def render(self, ctx):
        ctx.set_source_rgba(*self.color)
        rounded_rect(ctx, *self.get_top_left(), *self.size, self.radius)
        ctx.fill()
        ctx.stroke()

        Container.render(self, ctx)


# ===| Utilities |===


def rounded_rect(ctx, x, y, width, height, radius):
    ctx.arc(x + radius, y + radius, radius, TAU / 2, 3 * TAU / 4)
    ctx.arc(x + width - radius, y + radius, radius, 3 * TAU / 4, TAU)
    ctx.arc(x + width - radius, y + height - radius, radius, 0, TAU / 4)
    ctx.arc(x + radius, y + height - radius, radius, TAU / 4, TAU / 2)
