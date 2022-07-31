#!/usr/bin/env python3.10
""""""
from render import *
import colour
import regex as re


class ModeEnum(enum.Enum):
    NORMAL = enum.auto()
    INSERT = enum.auto()
    SHELL = enum.auto()
    NIMPL0 = enum.auto()
    NIMPL1 = enum.auto()


@dc.dataclass
class Mode:
    mode: str
    color: list

    def __post_init__(self):
        self.color = utils.normalize_color(self.color)
        self.color = [i for i in self.color][:3]

        # Normalize
        self.color = colour.rgb2hsl(self.color)
        self.color = list(self.color)
        self.color[-1] = 0.5
        self.color = colour.hsl2rgb(self.color)
        self.color = np.array([*self.color, 1])


NORMAL = ModeEnum.NORMAL
INSERT = ModeEnum.INSERT
SHELL = ModeEnum.SHELL
NIMPL0 = ModeEnum.NIMPL0
NIMPL1 = ModeEnum.NIMPL1

MODES = {
    NORMAL: Mode(NORMAL, "#224870"),
    INSERT: Mode(INSERT, "#1B998B"),
    SHELL: Mode(SHELL, "#5EE08C"),
    NIMPL0: Mode(..., "#FBA823"),
    NIMPL1: Mode(..., "#BA1B1D"),
}


# ===| Globals |===

with open("interpolate.py", "r", encoding="utf-8") as file:
    DEFAULT_TEXT = file.read()

# ===| Classes |===

# Stuff for the terminalesque aesthetic
@dc.dataclass
class Editor(Container):
    buffer: str = DEFAULT_TEXT
    pointer: int = 0

    body: Container = dc.field(default_factory=Container)
    current_mode: Mode = MODES[NORMAL]
    top_info: RoundedRectangle = dc.field(default_factory=RoundedRectangle)
    mode_indicator: RoundedRectangle = dc.field(default_factory=RoundedRectangle)
    top_line: int = 0

    inner_spacing: int = 16
    text_spacing: int = 10

    font_size: int = 20
    char_ratio: float = 0.5  # The x/y (DPI) font ratio for Iosevka.

    def __post_init__(self):
        for i in self.body, self.top_info, self.mode_indicator:
            i.parent = self
            i.color = utils.Color.ACCENT_0
            i.anchor = Anchor.TOP | Anchor.LEFT

        self.mode_indicator.children.append(
            Text(
                text=self.current_mode.mode.name,
                font_size=self.font_size,
                font_weight=cairo.FONT_WEIGHT_BOLD,
            )
        )

        for i in self.mode_indicator.children:
            i.parent = self.mode_indicator

    def render(self, ctx):
        # TODO: draw bottom bgcolor rectangle which hides potential part offscreen numbers
        # TODO: moldy code unless fixed

        spacing = self.inner_spacing
        top_left = self.get_top_left()

        # Place the top info box.
        self.top_info.position = top_left + spacing
        self.top_info.size = np.array(
            [self.size[0] - spacing * 2, self.font_size + self.text_spacing * 2]
        )

        # Place the bottom info box.
        self.mode_indicator.anchor = Anchor.BOTTOM | Anchor.LEFT
        self.mode_indicator.position = (
            top_left + [0, self.size[1]] + [spacing, -spacing]
        )
        self.mode_indicator.size = [
            (self.font_size / 2 * 6) + self.text_spacing * 2,
            self.top_info.size[1],
        ]
        self.mode_indicator.color = self.current_mode.color

        # Set the text box positions.
        for child in self.mode_indicator.children:
            child.position = (
                child.parent.get_top_left()
                + np.array(child.parent.size) * 0.5
                + [0, self.text_spacing]
            )
            child.anchor = Anchor.TOP
            child.font_size = self.font_size

        # Place the body.
        available_y_space = self.size[1] - (
            self.mode_indicator.size[1] + self.top_info.size[1] + spacing * 2
        )
        self.body.position = (
            top_left + [spacing, 2 * spacing] + [0, self.top_info.size[1]]
        )
        self.body.size = np.array(
            [
                self.size[0] - spacing * 2,
                available_y_space - spacing * 2,
            ]
        )

        self._update_body(ctx)

        self.top_info.render(ctx)
        self.mode_indicator.render(ctx)
        self.body.render(ctx)

    def switch_mode(self, new_mode):
        self.current_mode = MODES[new_mode]
        self.mode_indicator.children[-1].text = self.current_mode.mode.name

    def move_pointer(self, steps, strict: bool = False):
        if isinstance(steps, (str, re.Pattern)):
            match = re.match(steps, self.buffer[self.pointer :])
            if match is None:
                assert not strict
            return self.move_pointer(len(match.groups(0)))

        assert isinstance(steps, int)
        modified = self.pointer + steps
        if modified in range(0, len(self.buffer)):
            self.pointer = modified
            return None

        assert not strict

    def handle_input(self, key: int, name: str):
        """Handle user input."""
        current = self.current_mode.mode

        # Switch back to Normal mode on escape.
        if name == "Escape" and current != NORMAL:
            self.switch_mode(NORMAL)
            return None

        if current == NORMAL:
            return self._handle_normal_mode(key, name)
        elif current == SHELL:
            return self._handle_shell_mode(key, name)
        elif current == INSERT:
            return self._handle_insert_mode(key, name)
        raise NotImplementedError()

    def _handle_normal_mode(self, key: int, name: str):
        """Handle normal mode."""
        if name == "i":
            self.switch_mode(INSERT)
        elif name == "s":
            self.switch_mode(SHELL)

    def _handle_shell_mode(self, key: int, name: str):
        """Handle shell mode."""

    def _handle_insert_mode(self, key: int, name: str):
        """Handle insert mode."""
        char = chr(key)

        if name == "Right":
            self.move_pointer(1)
        if name == "Left":
            self.move_pointer(-1)
        elif char != "\x00":
            self.buffer = (
                self.buffer[: self.pointer] + char + self.buffer[self.pointer :]
            )

    def _get_body_char_size(self, ctx):
        txt = Text()
        ctx.select_font_face(txt.font_face, txt.font_slant, txt.font_weight)
        ctx.set_font_size(self.font_size)
        (_, _, w, h, _, _) = ctx.text_extents("#")

        return (self.body.size) / [w, h + self.text_spacing]

    def _update_body(self, ctx):
        LJUST_AMT = 8
        queue = self.buffer.splitlines()
        queue = [f"{str(i).ljust(LJUST_AMT)}{txt}" for i, txt in enumerate(queue)]

        self.body.children = []
        char_size = self._get_body_char_size(ctx)
        char_size = list(map(int, char_size))
        char_size = [i - 2 for i in char_size]
        assert all(i > 10 for i in char_size)

        i = 0
        while queue and (i < char_size[1]):
            line = queue.pop(0)
            if len(line) > char_size[0]:
                line, rest = line[: char_size[0]], line[char_size[0] :]
                queue.insert(0, "·".ljust(LJUST_AMT) + rest)

            self.body.children.append(
                Text(
                    text=line,
                    color=utils.Color.WHITE,
                    font_size=self.font_size,
                    position=self.body.get_top_left()
                    + self.text_spacing
                    + [0, 2 * self.font_size]
                    + [0, i * (self.text_spacing + self.font_size)],
                    anchor=Anchor.BOTTOM | Anchor.LEFT,
                )
            )

            i += 1
