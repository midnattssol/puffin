#!/usr/bin/env python3
"""Run Puffin."""
import dataclasses as dc
import math
import time
from datetime import datetime
import typing as t

import cairo
import gi
import numpy as np
from editor import Editor

from shortcake import *
import shortcake as sc
import cairo

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
    WIDGETS = dict(
        editor=editor,
    )

    WIDGETS["editor"].kill = kill


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
    name = Gdk.keyval_name(val)

    sc.HELD_DOWN[name] = True
    WIDGETS["editor"].handle_input(Gdk.keyval_to_unicode(val), Gdk.keyval_name(val))


def release_callback(window, key):
    """Callback for button releases."""
    val = key.keyval
    name = Gdk.keyval_name(val)

    sc.HELD_DOWN[name] = False


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
    colors = {
        "ACCENT_0": "#8F1445",
        "ACCENT_1": "#AE375B",
        "ACCENT_2": "#C86574",
        "ACCENT_3": "#E6B2BB",
        "WHITE": "#FCFCFF",
        "GREY_L1": "#5B586A",
        "GREY": "#4B4957",
        "GREY_D1": "#3A3943",
        "GREY_D2": "#29292F",
        "BLACK": "#18191B",
    }

    for k, v in colors.items():
        setattr(utils.Color, k, utils.normalize_color(v))

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
    win.connect("key-release-event", release_callback)
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
