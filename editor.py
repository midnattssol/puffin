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


# Stuff for the terminalesque aesthetic
@dc.dataclass
class Editor(Container):
    buffer: str = DEFAULT_TEXT
    pointer: int = 1
    focused_line: int = 0

    body: Container = dc.field(default_factory=Container)
    current_mode: Mode = MODES[NORMAL]
    top_info: RoundedRectangle = dc.field(default_factory=RoundedRectangle)
    mode_indicator: RoundedRectangle = dc.field(default_factory=RoundedRectangle)

    focused_line: int = 0
    top_line: int = 0
    inner_spacing: int = 16

    font_size: int = 20
    char_ratio: float = 0.5  # The x/y (DPI) font ratio for Iosevka.

    cursor: Rectangle = dc.field(default_factory=Rectangle)

    # buffer_manager: TextGridManager = dc.field(default_factory=TextGridManager)
    # split_buffer: list = None

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

        self.top_info.render(ctx)
        self.mode_indicator.render(ctx)
        self.body.render(ctx)

        # Render the pointer.
        # self._render_pointer(ctx)

    def switch_mode(self, new_mode: ModeEnum) -> None:
        """Switch the current mode.."""
        self.current_mode = MODES[new_mode]
        self.mode_indicator.children[-1].text = self.current_mode.mode.name

    def set_pointer(self, pos: int, strict: bool = False) -> None:
        """Set the position of the pointer."""
        steps = pos - self.pointer
        return self.move_pointer(steps, strict=strict)

    def move_pointer(self, steps, strict: bool = False) -> None:
        """Move the pointer forwards a number of steps or into the next match of some pattern."""
        if isinstance(steps, (str, re.Pattern)):
            match = re.match(steps, self.buffer[self.pointer :])
            if match is None:
                assert not strict
            return self.move_pointer(len(match.groups(0)))

        assert isinstance(steps, int)
        modified = self.pointer + steps
        if modified in range(1, len(self.buffer)):
            self.pointer = modified
            return None

        assert not strict
        self.pointer = max(1, min(len(self.buffer) - 1, self.pointer))
        return None

    def handle_input(self, key: int, name: str) -> None:
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
        elif name == "h":
            # Go backwards one character.
            self.move_pointer(-1)
        elif name == "l":
            # Go forwards one character.
            self.move_pointer(1)
        elif name == "j":
            # Jump backwards one line.
            self.set_pointer(line_num2index(-1, self.pointer, self.buffer) + 2)
        elif name == "k":
            # Jump forwards one line.
            self.set_pointer(line_num2index(1, self.pointer, self.buffer) + 2)

    def _handle_shell_mode(self, key: int, name: str):
        """Handle shell mode."""

    def _handle_insert_mode(self, key: int, name: str):
        """Handle insert mode."""
        char = chr(key)

        # print(name, key, char)

        if name == "Left":
            # Go backwards one character.
            self.move_pointer(-1)
        elif name == "Right":
            # Go forwards one character.
            self.move_pointer(1)
        elif name == "Up":
            # Jump backwards one line.
            self.set_pointer(line_num2index(-1, self.pointer, self.buffer) + 2)
        elif name == "Down":
            # Jump forwards one line.
            self.set_pointer(line_num2index(1, self.pointer, self.buffer) + 2)
        elif name == "Delete":
            self.buffer = self.buffer[: self.pointer - 1] + self.buffer[self.pointer :]

        elif name == "BackSpace":
            self.buffer = (
                self.buffer[: self.pointer - 2] + self.buffer[self.pointer - 1 :]
            )
            self.move_pointer(-1)

        elif char != "\x00":
            # Input the character.
            if char == "\r":
                char = "\n"
            self.buffer = (
                self.buffer[: self.pointer - 1] + char + self.buffer[self.pointer - 1 :]
            )
            self.move_pointer(1)

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
        if self.current_mode.mode == NORMAL:
            self.cursor = RoundedRectangle(
                size=[
                    self._get_char_size(ctx)[0] * 1.2,
                    self._get_char_size(ctx)[1] * 1.2,
                ],
                color=utils.Color.ACCENT_0,
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
        offset_pointer = self.pointer - 1

        char_size = self._get_body_n_chars(ctx)
        char_size = list(map(int, char_size))
        char_size[0] -= 2
        assert all(i > 10 for i in char_size)

        # Get the lines.
        queue = []
        lines = self.buffer.splitlines(keepends=True)
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

            self.body.children.append(
                Text(
                    text=line.removesuffix("\n"),
                    color=utils.Color.WHITE,
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
                self.cursor.render(ctx)
                cursor_drawn = True

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
                selector.render(ctx)

            n_drawn_lines += 1

    def _focus_selected_line(self, ctx) -> None:
        """Makes sure that the selected line is focused."""
        lines = self.buffer.splitlines(keepends=True)
        acc = 0
        i = 0
        for i, line in enumerate(lines):
            acc += len(line)
            if acc >= self.pointer:
                break

        self.top_line = i

        char_size = self._get_body_n_chars(ctx)
        lines_on_screen = int(char_size[1])

        top_line_to_show = i - (lines_on_screen // 2)
        top_line_to_show = max(0, top_line_to_show)
        self.focused_line = top_line_to_show
