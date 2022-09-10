"""Enumerations."""

import enum

# ===| Enumerations |===


class BufferAction(enum.Enum):
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
    NORMAL = enum.auto()
    INSERT = enum.auto()
    SHELL = enum.auto()
    NIMPL0 = enum.auto()
    NIMPL1 = enum.auto()
