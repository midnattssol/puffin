"""Enumerations."""

import enum

# ===| Enumerations |===


class BufferAction(enum.Enum):
    """An action which text buffers can handle."""

    DELETE = enum.auto()
    YANK = enum.auto()
    PASTE = enum.auto()
    GO = enum.auto()
    LEFT = enum.auto()
    DOWN = enum.auto()
    UP = enum.auto()
    RIGHT = enum.auto()
    WORD = enum.auto()


class ModeEnum(enum.Enum):
    """Modes that the text editor can be in."""

    NORMAL = enum.auto()
    INSERT = enum.auto()
    SHELL = enum.auto()
    NIMPL0 = enum.auto()
    NIMPL1 = enum.auto()
