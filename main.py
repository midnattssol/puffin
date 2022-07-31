#!/usr/bin/env python3
"""A skeleton widget."""
import dataclasses as dc
import math
import time
from datetime import datetime
import typing as t

import cairo
import gi
import interpolate as ip
import numpy as np
import render
import size
import utils
from enums import *
from editor import Editor

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, GLib

# ===| Globals |===

WINDOW = None
FPS = 60
SIZE = np.array([300, 600])
WHITE = utils.normalize_color("#f0f0f0")
BLACK = utils.normalize_color("#142424")

TIME_TO_KILL_AT = None

WIDGETS = {}

# A top container.
def get_widgets():
    global WIDGETS
    editor = Editor(position=np.array([0, 0]), anchor=Anchor.TOP | Anchor.LEFT)
    # border = render.RoundedRectangle(color=utils.Color.WHITE)
    WIDGETS = dict(
        # border=border,
        editor=editor,
    )


text = """
def get_widgets():
    global WIDGETS
    rectangle = render.RoundedRectangle(
        position=size.TimeFunction(
            ip.oscillating(
                SIZE / 2 - [0, 100],
                SIZE / 2 + [0, 100],
            )
        ),
        size=[200, 100],
        direction=Direction.VERTICAL,
""".splitlines()


def draw(da, ctx):
    """Draw everything that should be drawn to the screen. Called once per frame."""
    if TIME_TO_KILL_AT is not None and TIME_TO_KILL_AT < time.time():
        exit()

    WIDGETS["editor"].size = np.array(WINDOW.get_size())

    for name, widget in WIDGETS.items():
        widget.render(ctx)


def kill(seconds_left=0):
    """Kill the widget in some time."""
    global TIME_TO_KILL_AT
    TIME_TO_KILL_AT = time.time() + seconds_left


# ===| Callbacks |===


def timeout_callback(widget):
    """Update the screen."""
    widget.queue_draw()
    return True


def press_callback(window, key):
    """Callback for button presses."""
    val = key.keyval

    if val == Gdk.KEY_Tab:
        kill()
    elif val == Gdk.KEY_plus:
        WIDGETS["editor"].font_size += 3
    elif val == Gdk.KEY_minus:
        WIDGETS["editor"].font_size -= 3
    else:
        WIDGETS["editor"].handle_input(Gdk.keyval_to_unicode(val), Gdk.keyval_name(val))


def click_callback(window, event):
    """Callback for mouse clicks."""
    # window.begin_move_drag(
    #     event.button,
    #     round(event.x + window.get_window().get_root_origin()[0]),
    #     round(event.y + window.get_window().get_root_origin()[1]),
    #     event.time,
    # )


def main():
    global WINDOW
    colors = [
        "#E7F7C9",
        "#101D24",
        "#468C61",
        "#133845",
    ]

    utils.Color.WHITE = utils.normalize_color(colors[0])
    utils.Color.BLACK = utils.normalize_color(colors[1])
    utils.Color.ACCENT_0 = utils.normalize_color(colors[2])
    utils.Color.ACCENT_1 = utils.normalize_color(colors[3])

    get_widgets()

    win = Gtk.Window(title="puffin-editor")
    win.move(0, 0)
    screen = win.get_screen()
    rgba = screen.get_rgba_visual()
    win.set_visual(rgba)
    win.add_events(Gdk.EventMask.BUTTON_PRESS_MASK)

    WINDOW = win

    # Connect callbacks.
    win.connect("key-press-event", press_callback)
    win.connect("destroy", lambda w: Gtk.main_quit())
    win.connect("button-press-event", click_callback)

    # Prepare for drawing.
    drawing_area = Gtk.DrawingArea()
    drawing_area.connect("draw", draw)
    GLib.timeout_add(1000 / FPS, timeout_callback, win)
    win.add(drawing_area)
    win.show_all()

    Gtk.main()


if __name__ == "__main__":
    main()
