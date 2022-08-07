#!/usr/bin/env python3.10
"""The editor."""
import dataclasses as dc
import enum
import itertools as it

import cairo
import colour
import more_itertools as mit
import movement as mov
import numpy as np
import regex as re
from shortcake import *

TAB_CHAR = " " * 4


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

with open(__file__, "r", encoding="utf-8") as file:
    DEFAULT_TEXT = file.read()
    DEFAULT_TEXT = DEFAULT_TEXT.splitlines()[:20]
    DEFAULT_TEXT = "\n".join(DEFAULT_TEXT)
    # DEFAULT_TEXT = "█" * 5 + file.read()

# ===| Classes |===


def line_num2index(n: int, pos: int, buf: str):
    def inner(pos, left):
        if left == 0:
            return pos, left
        if n < 0:
            _buf = buf[: max(0, pos - 2)]
            idx = _buf.rfind("\n")
            if idx == -1:
                return (-1, None)
        if n > 0:
            _buf = buf[pos - 1 :]
            idx = _buf.find("\n")
            if idx == -1:
                return (len(buf) - 1, None)
            idx += pos - 1

        return inner(idx, left - 1)

    left = abs(n)
    result, _ = inner(pos, left)
    return result


# @dc.dataclass
# class TextGridManager:
#     """Automatically calculates certain values of a text grid on update."""
#
#     abs_size: tuple = None
#     char_size: tuple = None
#     grid_size: tuple = None
#     text_spacing: int = None
#     font_size: int = 20
#
#     def __post_init__(self):
#         self._update_char_size()
#         self._update_grid_size()
#
#     def get_pos(self, x, y):
#         x_pos = x * self.char_size[0]
#         y_pos = (
#             self.text_spacing
#             + 2 * self.font_size
#             + y * (self.text_spacing + self.char_size[1])
#         )
#         return np.array([x_pos, y_pos])
#
#     def set_font_size(self, n):
#         self.font_size = n
#         self._update_text_spacing()
#
#     def resize(self, ctx, abs_size):
#         self.abs_size = abs_size
#         self._update_char_size(ctx)
#         self._update_grid_size(ctx)
#
#     # Update functions
#     def _update_char_size(self):
#         txt = Text()
#         utils.CONTEXT.select_font_face(txt.font_face, txt.font_slant, txt.font_weight)
#         utils.CONTEXT.set_font_size(self.font_size)
#         (_, _, w, h, _, _) = ctx.text_extents("#")
#
#         self.char_size = np.array([w, h])
#
#     def _update_grid_size(self):
#         w, h = self._get_char_size(utils.CONTEXT)
#
#     def _update_text_spacing(self):
#         self.text_spacing = self.font_size * 0.3


NOUNS = r"[hjklw]"
SCALE = r"(?<scale>-?\d+)"

SHORT = {
    "d": "delete",
    "p": "paste",
    "g": "go",
    "h": "left",
    "j": "down",
    "k": "up",
    "l": "right",
    "w": "word",
}

FORMS = {
    "delete": rf"(?<act>d){SCALE}?(?<noun>[d[{{('\"]|{NOUNS})",
    "paste": rf"(?<act>p)",
    "go": rf"(?<act>g){SCALE}?",
    "left": rf"{SCALE}?(?<act>h)",
    "down": rf"{SCALE}?(?<act>j)",
    "up": rf"{SCALE}?(?<act>k)",
    "right": rf"{SCALE}?(?<act>l)",
    "word": rf"{SCALE}?(?<act>w)",
}


@dc.dataclass
class NormalModeHandler:
    read_buffer: str = ""
    write_buffer: str = ""
    registers: dict = dc.field(default_factory=lambda: {"copy": ""})

    def command_is_finished(self):
        """Check if a command should be executed immediately without waiting for the user to click Enter."""
        return self.parse_command() is not None

    def parse_command(self):
        """Parse a command."""
        name = form = match = None

        # Try all forms.
        for name, form in FORMS.items():
            match = re.search(form, self.read_buffer.text)
            if match:
                break

        # Make sure a match was made.
        if (not match) or (len(match.group()) != len(self.read_buffer.text)):
            # raise NotImplementedError(match)
            return None

        # print(name, form, match)
        groups = match.groupdict()

        # TODO: Handle weird scales here like dd and similar
        # TODO: Handle negative numbers

        groups["scale"] = 1 if groups.get("scale") is None else int(groups["scale"])
        assert groups["scale"] >= 0
        self.read_buffer.clear()

        if name == "left":
            action = BufferAction.LEFT
        elif name == "right":
            action = BufferAction.RIGHT
        elif name == "word":
            action = BufferAction.WORD
        elif name == "up":
            action = BufferAction.UP
        elif name == "paste":
            for _ in range(groups["scale"]):
                self.write_buffer.write(
                    self.write_buffer.pointer, self.registers["copy"]
                )
            return None
        elif name == "down":
            action = BufferAction.DOWN
        elif name == "delete":
            action = BufferAction[SHORT[groups["noun"]].upper()]
            old_pos = self.write_buffer.pointer

            for _ in range(groups["scale"]):
                self.write_buffer.handle_buffer_action(action)

            new_pos = self.write_buffer.pointer
            old_pos, new_pos = sorted([old_pos, new_pos])

            middle = self.write_buffer.text[old_pos - 1 : new_pos - 1]
            self.registers["copy"] = middle

            self.write_buffer.text = (
                self.write_buffer.text[: old_pos - 1]
                + self.write_buffer.text[new_pos - 1 :]
            )
            self.write_buffer.set_pointer(old_pos)
            return None
        else:
            raise NotImplementedError()

        self.read_buffer.clear()

        for _ in range(groups["scale"]):
            self.write_buffer.handle_buffer_action(action)


@dc.dataclass
class TextBuffer:
    text: str = "Hello world"
    default: str = "..."
    pointer: int = 1
    multiline: bool = True
    newline_callback: callable = lambda self: None
    write_callback: callable = lambda self: None
    vertical_movement_pointer: int = 0

    def __bool__(self):
        return bool(self.text)

    def reset_v_pointer(self):
        self.vertical_movement_pointer = -1

    def move_pointer(self, steps, strict: bool = False) -> None:
        """Move the pointer forwards a number of steps or into the next match of some pattern."""
        if isinstance(steps, (str, re.Pattern)):
            match = re.match(steps, self.text[self.pointer :])
            if match is None:
                assert not strict
            return self.move_pointer(len(match.groups(0)))

        assert isinstance(steps, int), (steps, type(steps))
        modified = self.pointer + steps
        if modified in range(1, len(self.text) + 2):
            self.pointer = modified
        else:
            assert not strict
            self.pointer = max(1, min(len(self.text), modified))

        dist_to_newline = self.text[: self.pointer - 1][::-1].find("\n")

        self.vertical_movement_pointer = max(
            self.vertical_movement_pointer,
            0,
            dist_to_newline if dist_to_newline != -1 else 0,
        )
        return None

    def set_pointer(self, pos: int, strict: bool = False) -> None:
        """Set the position of the pointer."""
        steps = pos - self.pointer
        return self.move_pointer(steps, strict=strict)

    def write(self, pos: int, chars: str, move_pointer: bool = True) -> None:
        self.text = self.text[:pos] + chars + self.text[pos:]
        self.write_callback(self)
        if move_pointer:
            self.pointer += len(chars)

    def delete(self, n):
        self.text = self.text[: max(0, self.pointer - n)] + self.text[self.pointer :]

    def send_text(self, key: int, name: str, buffer_override=None) -> None:
        """Handles sending text and similar modifications to the selected buffer."""
        char = chr(key)

        mapping = {"Tab": TAB_CHAR}

        if name in mapping:
            char = mapping[name]

        # Special characters.
        if name == "ISO_Left_Tab":
            line_index = line_num2index(0, self.pointer, self.text) + 1
            offset_to_newline = (
                self.pointer
                if line_index == 0
                else (self.text[: self.pointer].rfind("\n") - self.pointer)
            )
            offset_to_newline = abs(offset_to_newline)

            line_index -= offset_to_newline
            line_index = max(0, line_index)
            line = self.text[line_index:].splitlines()[0]
            original_line_length = len(line)

            if line.startswith(TAB_CHAR):
                line = line.removeprefix(TAB_CHAR)
            else:
                line = line.lstrip()

            # Actually make the move.
            self.move_pointer(len(line) - original_line_length)
            self.text = (
                self.text[:line_index:]
                + line
                + self.text[line_index + original_line_length :]
            )

        elif name == "Left":
            # Go backwards one character.
            self.move_pointer(-1)
        elif name == "Right":
            # Go forwards one character.
            self.move_pointer(1)
        elif name == "Up":
            # Jump backwards one line.
            self.set_pointer(line_num2index(-1, self.pointer, self.text) - 2)
        elif name == "Down":
            # Jump forwards one line.
            self.set_pointer(mov.nxt_line_start(self.pointer, self.text))
        elif name == "Delete":
            self.delete(1)
        elif name == "BackSpace":
            self.move_pointer(-1)
            self.delete(1)

        elif char != "\x00":
            # Input the character.
            if char == "\r":
                if not self.multiline:
                    return self.newline_callback(self)
                char = "\n"

            self.write(max(0, self.pointer - 1), char)

    def clear(self):
        self.pointer = 1
        self.text = ""

    def prev(self, char):
        return mov.prv_template(char)(self.pointer, self.text)

    def next(self, char):
        return (
            mov.nxt_template(
                char,
            )(self.pointer, self.text)
            + 1
            - (
                (self.pointer - 1) in range(len(self.text))
                and (self.text[self.pointer - 1] == "\n")
            )
        )

    def current_line(self):
        return self.text[: self.pointer].count("\n")

    def handle_buffer_action(self, action: BufferAction):
        """Handle actions."""
        if action == BufferAction.LEFT:
            # Go backwards one character.
            self.reset_v_pointer()
            self.move_pointer(-1)
        elif action == BufferAction.RIGHT:
            # Go forwards one character.
            self.reset_v_pointer()
            self.move_pointer(1)
        elif action == BufferAction.WORD:
            self.reset_v_pointer()
            pos = self.next(r"(.|\n)(\b|\n)\w")
            self.set_pointer(pos)
        elif action in {BufferAction.UP, BufferAction.DOWN}:
            if action == BufferAction.UP:
                # Go back to the top if on first line.
                if self.current_line() == 0:
                    self.reset_v_pointer()
                    self.set_pointer(1)
                    return None
                pos = self.prev(r"[^\n]*\n")
            elif action == BufferAction.DOWN:
                # Go to the end of the buffer if on last line.
                if self.current_line() == self.text.count("\n"):
                    self.reset_v_pointer()
                    self.set_pointer(int(1e20))
                    return None

                pos = self.next(r"\n")
            else:
                raise NotImplementedError()

            line = self.text[:pos].count("\n")

            # Move to the same location in the next line.
            lines = self.text.splitlines(keepends=True)
            endl_0 = sum(map(len, lines[:line]))
            endl_1 = sum(map(len, lines[: line + 1]))
            pos = endl_0 + self.vertical_movement_pointer + 1

            # Make sure the position is on the right line.
            pos = min(endl_1, pos)
            self.set_pointer(pos)
        else:
            raise NotImplementedError()


# Stuff for the terminalesque aesthetic
@dc.dataclass
class Editor(Container):
    normal_handler: NormalModeHandler = dc.field(default_factory=NormalModeHandler)
    body_buffer: TextBuffer = dc.field(default_factory=lambda: TextBuffer(DEFAULT_TEXT))
    normal_buffer: TextBuffer = dc.field(
        default_factory=lambda: TextBuffer(text="", multiline=False)
    )
    shell_buffer: TextBuffer = dc.field(
        default_factory=lambda: TextBuffer(text="", multiline=False)
    )
    active_buffer: ... = None

    focused_line: int = 0

    body: Container = dc.field(default_factory=Container)
    current_mode: Mode = MODES[NORMAL]
    top_info: RoundedRectangle = dc.field(default_factory=RoundedRectangle)
    mode_indicator: RoundedRectangle = dc.field(default_factory=RoundedRectangle)
    mode_buffer_container: RoundedRectangle = dc.field(default_factory=RoundedRectangle)

    focused_line: int = 0
    top_line: int = 0
    inner_spacing: int = 16

    font_size: int = 20
    char_ratio: float = 0.5  # The x/y (DPI) font ratio for Iosevka.

    cursor: Rectangle = dc.field(default_factory=Rectangle)

    def __post_init__(self):
        self.active_buffer = self.body_buffer
        self.shell_buffer.newline_callback = lambda s: self._handle_shell_command(
            s.text
        )
        # self.normal_buffer.write_callback = lambda s: self._handle_shell_command(s.text)

        for i in (
            self.body,
            self.top_info,
            self.mode_indicator,
            self.mode_buffer_container,
        ):
            i.parent = self
            i.color = utils.Color.ACCENT_0
            i.anchor = Anchor.TOP | Anchor.LEFT

        self.mode_buffer_container.color = utils.Color.GREY_D1

        x = len(self.current_mode.mode.name)

        self.mode_indicator.children.append(
            Text(
                text=self.current_mode.mode.name,
                font_size=self.font_size,
                font_weight=cairo.FONT_WEIGHT_BOLD,
            )
        )

        self.mode_buffer_container.children.append(
            Text(
                text="Hello world!",
                font_size=self.font_size,
            )
        )

        for i in self.mode_indicator.children:
            i.parent = self.mode_indicator
        for i in self.mode_buffer_container.children:
            i.parent = self.mode_buffer_container

    def render(self, ctx):
        # TODO: draw bottom bgcolor rectangle which hides potential part offscreen numbers
        # TODO: moldy code unless fixed

        spacing = self.inner_spacing
        top_left = self.get_top_left()

        # Place the top info box.
        self.top_info.position = top_left + spacing
        self.top_info.size = np.array(
            [
                self.size[0] - spacing * 2,
                self.font_size + self._get_text_v_spacing() * 2,
            ]
        )

        # Place the bottom info box.
        self.mode_indicator.anchor = Anchor.BOTTOM | Anchor.LEFT
        self.mode_indicator.position = (
            top_left + [0, self.size[1]] + [spacing, -spacing]
        )
        self.mode_indicator.size = [
            (self.font_size / 2 * 6) + self._get_text_v_spacing() * 2,
            self.top_info.size[1],
        ]
        self.mode_indicator.color = self.current_mode.color

        self.mode_buffer_container.position = np.array(
            [
                self.mode_indicator.position[0] + self.mode_indicator.size[0] + spacing,
                self.mode_indicator.position[1],
            ]
        )

        self.mode_buffer_container.size = np.array(
            [
                self.size[0] - spacing - self.mode_buffer_container.position[0],
                self.mode_indicator.size[1],
            ]
        )

        self.mode_buffer_container.anchor = Anchor.BOTTOM | Anchor.LEFT

        # Set the text box positions.
        for child in self.mode_buffer_container.children:
            child.position = (
                child.parent.get_top_left()
                + np.array(child.parent.size) * [0, 0.5]
                + self._get_text_v_spacing()
            )
            child.anchor = Anchor.TOP | Anchor.LEFT
            child.color = utils.Color.WHITE
            child.font_size = self.font_size
            if self.current_mode.mode == NORMAL:
                child.text = self.normal_buffer.text
            elif self.current_mode.mode == SHELL:
                child.text = self.shell_buffer.text
            else:
                child.text = ""

        # Set the text box positions.
        for child in self.mode_indicator.children:
            child.position = (
                child.parent.get_top_left()
                + np.array(child.parent.size) * 0.5
                + [0, self._get_text_v_spacing()]
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

        # Do the rendering.
        self.top_info.render(ctx)
        self.mode_indicator.render(ctx)
        self.mode_buffer_container.render(ctx)
        self.body.render(ctx)

    def switch_mode(self, new_mode: ModeEnum) -> None:
        """Switch the current mode.."""
        self.current_mode = MODES[new_mode]
        self.mode_indicator.children[-1].text = self.current_mode.mode.name

        if new_mode == SHELL:
            self.active_buffer = self.shell_buffer
        # elif new_mode == NORMAL:
        #     self.active_buffer = self.normal_buffer
        else:
            self.active_buffer = self.body_buffer

    def handle_input(self, key: int, name: str) -> None:
        """Handle user input."""
        current = self.current_mode.mode

        # Switch back to Normal mode on escape.
        if name == "Escape" and current != NORMAL:
            self.switch_mode(NORMAL)
            return None

        if name == "plus":
            self.font_size += 3
            return
        elif name == "minus":
            self.font_size -= 3
            self.font_size = max(5, self.font_size)
            return
        elif current == NORMAL:
            return self._handle_normal_mode(key, name)
        elif current in {SHELL, INSERT}:
            return self.active_buffer.send_text(key, name)
        raise NotImplementedError()

    def _handle_normal_mode(self, key: int, name: str):
        """Handle normal mode."""
        self.normal_buffer.send_text(key, name)

        buffer = self.active_buffer
        self.normal_handler.read_buffer = self.normal_buffer
        self.normal_handler.write_buffer = self.active_buffer

        if name == "colon":
            self.switch_mode(SHELL)
            self.normal_buffer.clear()
        elif name == "i":
            self.switch_mode(INSERT)
            self.normal_buffer.clear()

        return self.normal_handler.parse_command()

    def _handle_shell_command(self, command):
        """Handles a shell command when it's entered."""
        command = command.strip()

        if command == "q":
            return self.kill()

        self.shell_buffer.text = ""

    def _get_text_v_spacing(self):
        """Get the vertical spacing between different characters."""
        return self.font_size * 0.3

    def _get_char_size(self, ctx) -> np.ndarray:
        """Get the size of a character in this font."""
        txt = Text()
        ctx.select_font_face(txt.font_face, txt.font_slant, txt.font_weight)
        ctx.set_font_size(self.font_size)
        (_, _, w, h, _, _) = ctx.text_extents("█")

        return np.array([w, h])

    def _get_body_n_chars(self, ctx) -> np.ndarray:
        """Get a ndarray that contains the number of rows and columns available in the body."""
        w, h = self._get_char_size(ctx)
        return (self.body.size) / [w, h + self._get_text_v_spacing()]

    def _get_pos(self, x, y, ctx):
        return self.body.get_top_left() + [
            x * self._get_char_size(ctx)[0],
            (
                y * (self._get_text_v_spacing() + self._get_char_size(ctx)[1])
                + self._get_text_v_spacing()
                + 2 * self.font_size
            ),
        ]

    def _update_cursor(self, ctx):
        if self.current_mode.mode != INSERT:
            self.cursor = RoundedRectangle(
                size=[
                    self._get_char_size(ctx)[0] * 1.2,
                    self._get_char_size(ctx)[1] * 1.2,
                ],
                color=utils.Color.ACCENT_1,
                radius=self._get_char_size(ctx)[0] * 0.3,
                anchor=Anchor.LEFT,
            )
        else:
            self.cursor = Rectangle(
                size=[
                    self._get_char_size(ctx)[0] * 0.2,
                    self._get_char_size(ctx)[1] * 1.5,
                ],
                color=utils.Color.WHITE,
                anchor=Anchor.LEFT,
            )

    def _update_body(self, ctx):
        self._update_cursor(ctx)
        self._focus_selected_line(ctx)

        LINE_NUM_CHARS = 10

        # Generate line number and line.
        def get_item(line, num=None):
            num = "·" if num is None else str(num + self.focused_line)
            num = num.rjust(5).ljust(LINE_NUM_CHARS)
            return num, line

        self.body.children = []
        cursor_drawn = False
        n_drawn_lines = 0
        offset_pointer = self.body_buffer.pointer - 1

        char_size = self._get_body_n_chars(ctx)
        char_size = list(map(int, char_size))
        char_size[0] -= 2
        assert all(i > 10 for i in char_size)

        # Get the lines.
        queue = []
        lines = self.body_buffer.text.splitlines(keepends=True)
        n_chars = sum(map(len, lines[: self.focused_line]))
        lines = lines[max(0, self.focused_line) :]

        for i, line in enumerate(lines):
            item = get_item(line, i)
            queue.append(item)

        # Draw as many lines as possible until the screen runs out of space.
        while queue and (n_drawn_lines < char_size[1]):
            num, line = queue.pop(0)
            d_len = len(num)

            # If the line is too long, it's broken up into two parts.
            if (len(line) + d_len) > char_size[0]:
                char_size_d = char_size[0] - d_len
                line, rest = line[:char_size_d], line[char_size_d:]
                queue.insert(0, get_item(rest))

            n_chars += len(line)

            # Add the children to be drawn.
            self.body.children.append(
                Text(
                    text=num,
                    color=utils.Color.GREY_L1,
                    font_size=self.font_size,
                    position=self._get_pos(0, n_drawn_lines, ctx),
                    anchor=Anchor.BOTTOM | Anchor.LEFT,
                )
            )

            x = len(line) - 1

            self.body.children.append(
                ColoredText(
                    text=line.removesuffix("\n"),
                    default_state=TextState(fore=utils.Color.WHITE),
                    font_size=self.font_size,
                    position=self._get_pos(LINE_NUM_CHARS, n_drawn_lines, ctx),
                    anchor=Anchor.BOTTOM | Anchor.LEFT,
                )
            )

            # Draw the pointer if it's on the right row of characters.
            if not cursor_drawn and n_chars > offset_pointer:
                x_coord = offset_pointer - n_chars + len(line)
                self.cursor.position = self._get_pos(
                    LINE_NUM_CHARS + x_coord, n_drawn_lines - 1, ctx
                )
                cursor_drawn = True

                if self.current_mode.mode != INSERT:
                    self.cursor.color = self.body.children[-1].get_state(x_coord).fore
                    self.body.children[-1].set_state(
                        x_coord, x_coord + 1, TextState(fore=utils.Color.BLACK)
                    )
                selector = RoundedRectangle(
                    size=[
                        self._get_char_size(ctx)[0] * 6,
                        self._get_char_size(ctx)[1] * 1.3,
                    ],
                    color=utils.Color.ACCENT_2,
                    radius=self._get_char_size(ctx)[0] * 0.3,
                    anchor=Anchor.LEFT,
                )

                self.body.children[-2].color = utils.Color.BLACK

                x_coord = offset_pointer - n_chars + len(line)
                selector.position = self._get_pos(0, n_drawn_lines - 1, ctx)

                # Do the rendering.
                self.cursor.render(ctx)
                selector.render(ctx)

            n_drawn_lines += 1

    def _focus_selected_line(self, ctx) -> None:
        """Makes sure that the selected line is focused."""
        lines = self.body_buffer.text.splitlines(keepends=True)
        acc = 0
        i = 0
        for i, line in enumerate(lines):
            acc += len(line)
            if acc >= self.body_buffer.pointer:
                break

        self.top_line = i

        char_size = self._get_body_n_chars(ctx)
        lines_on_screen = int(char_size[1])

        top_line_to_show = i - (lines_on_screen // 2)
        top_line_to_show = max(0, top_line_to_show)
        self.focused_line = top_line_to_show
