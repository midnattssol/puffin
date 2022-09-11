#!/usr/bin/env python3.10
"""The editor."""
import dataclasses as dc
import typing as t

import cairo
import colour
import movement as mov
import numpy as np
import regex as re
import shortcake as sc
import utils
from enums import *


@dc.dataclass
class ModeAndColor:
    """A mode that the text editor can be in."""

    mode: str
    color: list

    def __post_init__(self):
        """Normalize the color."""
        self.color = sc.utils.normalize_color(self.color)
        self.color = self.color[:3]

        # Normalize the color.
        self.color = colour.rgb2hsl(self.color)
        self.color = list(self.color)
        self.color[-1] = 0.5
        self.color = colour.hsl2rgb(self.color)
        self.color = np.array([*self.color, 1])


# ===| Globals |===


with open(__file__, "r", encoding="utf-8") as file:
    DEFAULT_TEXT = file.read()
    DEFAULT_TEXT = DEFAULT_TEXT.splitlines()[:20]
    DEFAULT_TEXT = "\n".join(DEFAULT_TEXT)
    DEFAULT_TEXT = DEFAULT_TEXT + "h" * 200

TAB_CHAR = " " * 4

NORMAL = ModeEnum.NORMAL
INSERT = ModeEnum.INSERT
SHELL = ModeEnum.SHELL
NIMPL0 = ModeEnum.NIMPL0
NIMPL1 = ModeEnum.NIMPL1
LINE_NUM_CHARS = 10

MODES = {
    NORMAL: ModeAndColor(NORMAL, "#224870"),
    INSERT: ModeAndColor(INSERT, "#1B998B"),
    SHELL: ModeAndColor(SHELL, "#5EE08C"),
    NIMPL0: ModeAndColor(..., "#FBA823"),
    NIMPL1: ModeAndColor(..., "#BA1B1D"),
}


NOUNS = r"[hjklw]"
SCALE = r"(?<scale>-?\d+)"

SHORT = {
    "d": "delete",
    "c": "cursor",
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
    "cursor": rf"(?<act>c){SCALE}?(?<noun>[d[{{('\"]|{NOUNS})",
    "paste": r"(?<act>p)",
    "go": rf"{SCALE}?(?<act>g)",
    "left": rf"{SCALE}?(?<act>h)",
    "down": rf"{SCALE}?(?<act>j)",
    "up": rf"{SCALE}?(?<act>k)",
    "right": rf"{SCALE}?(?<act>l)",
    "word": rf"{SCALE}?(?<act>w)",
}

assert set(SHORT.values()) == set(FORMS.keys()), set(SHORT.values()) - set(
    FORMS.keys()
) | set(FORMS.keys()) - set(SHORT.values())


# ===| Classes |===


@dc.dataclass
class TextBuffer:
    """A buffer object for text that provides cursors and modifications to the text using these cursors."""

    text: str = "Hello world!"
    multiline: bool = True
    newline_callback: callable = lambda self: None
    write_callback: callable = lambda self: None
    cursors: t.List = dc.field(default_factory=lambda: [Cursor()])

    def __post_init__(self):
        """Make sure that all child cursors have this buffer set as their parent."""
        for cursor in self.cursors:
            cursor.parent = self

    def line_bounds(self, num: int) -> t.Tuple[int]:
        """Get the line boundaries of some line."""
        lines = self.text.splitlines(keepends=True)
        num = min(len(lines) - 1, num)

        old_pos = sum(map(len, lines[:num]))
        new_pos = old_pos + len(lines[num])

        return old_pos, new_pos

    def write_at_all_pointers(self, chars: str, move_pointer: bool = True):
        """Add text at all pointers."""
        result = []

        for cursor in sorted(self.cursors, reverse=True):
            result.append(cursor.write(self, cursor.pointer, chars, move_pointer))

        return result

    def remove_all_but_one_cursor(self, new_pos=None):
        """Remove all except the first cursor."""
        self.cursors = self.cursors[:1]
        if new_pos is not None:
            self.cursors[0].move_pointer_and_siblings(new_pos)

    def remove_duplicate_cursors(self):
        """Remove duplicate cursors."""
        self.cursors.sort()
        prev = None
        i = 0

        # Remove pointers which are duplicates by checking if the
        # previous pointer was the same as the current one.
        # This works since the list is sorted.

        while i < len(self.cursors):
            cursor = self.cursors[i]

            if cursor.pointer == prev:
                self.cursors.pop(i)
                continue

            i += 1
            prev = cursor.pointer

    def clear(self):
        """Clear the text and merge all but one cursor."""
        self.cursors = self.cursors[:1]
        self.cursors[0].set_pointer(1)
        self.text = ""

    def send_text(self, key: int, name: str) -> None:
        """Send text to the cursor."""
        result = []

        for cursor in sorted(self.cursors, reverse=True):
            result.append(cursor.send_text(key, name))

        return result

    def handle_buffer_action(self, action: BufferAction) -> None:
        """Send a buffer action to the cursors."""
        result = []
        for cursor in sorted(self.cursors):
            result.append(cursor.handle_buffer_action(action))
        return result

    def reset_v_pointer(self) -> None:
        """Reset all vertical movement pointers."""
        result = []
        for cursor in sorted(self.cursors):
            result.append(cursor.reset_v_pointer())
        return result


@dc.dataclass
class NormalModeHandler:
    """Handles input to the text editor in Normal mode."""

    read_buffer: TextBuffer = dc.field(default_factory=TextBuffer)
    write_buffer: TextBuffer = dc.field(default_factory=TextBuffer)
    registers: dict = dc.field(default_factory=lambda: {"copy": ""})

    def command_is_finished(self) -> bool:
        """Check if a command is done."""
        return self.parse_command() is not None

    def parse_command(self) -> None:
        """Parse a command."""
        name = form = match = None

        # Try all forms.
        for name, form in FORMS.items():
            match = re.search(form, self.read_buffer.text)
            if match:
                break

        # Make sure a match was made.
        if (not match) or (len(match.group()) != len(self.read_buffer.text)):
            return

        groups = match.groupdict()

        # Default on weird scales.
        scale_default = 1
        if name == "go":
            scale_default = 0

        groups["scale"] = (
            scale_default if groups.get("scale") is None else int(groups["scale"])
        )

        assert groups["scale"] >= 0 or name == "go"
        self.read_buffer.clear()

        if name == "go":
            new_pos = self.write_buffer.line_bounds(groups["scale"])[0] + 1
            self.write_buffer.remove_all_but_one_cursor(new_pos=new_pos)
            return
        if name == "paste":
            for _ in range(groups["scale"]):
                self.write_buffer.write_at_all_pointers(self.registers["copy"])
            return
        if name == "cursor":
            action = BufferAction[SHORT[groups["noun"]].upper()]
            cursor = max(self.write_buffer.cursors, key=lambda x: x.pointer)
            cursor_copy = dc.replace(cursor)

            old_pos = cursor.pointer
            for _ in range(groups["scale"]):
                cursor.handle_buffer_action(action)
            new_pos = cursor.pointer
            old_pos, new_pos = sorted([old_pos, new_pos])

            self.write_buffer.cursors.append(cursor_copy)
            return
        if name == "delete":
            self._handle_delete(groups)
            return

        if name == "left":
            action = BufferAction.LEFT
        elif name == "right":
            action = BufferAction.RIGHT
        elif name == "word":
            action = BufferAction.WORD
        elif name == "up":
            action = BufferAction.UP
        elif name == "down":
            action = BufferAction.DOWN
        else:
            raise NotImplementedError()

        self.read_buffer.clear()

        for _ in range(groups["scale"]):
            self.write_buffer.handle_buffer_action(action)

    def _handle_delete(self, groups):
        action = BufferAction[SHORT[groups["noun"]].upper()]

        # Ranges to be deleted.
        ranges = []

        for cursor in self.write_buffer.cursors:

            if action == BufferAction.DELETE:
                # Handle deleting current line with dd
                cursor.reset_v_pointer()
                old_pos, new_pos = cursor.line_bounds()
                old_pos += 1
                new_pos += 1
            else:
                old_pos = cursor.pointer
                for _ in range(groups["scale"]):
                    cursor.handle_buffer_action(action)
                new_pos = cursor.pointer
                old_pos, new_pos = sorted([old_pos, new_pos])

            ranges.append((old_pos, new_pos))

        # Generate all the ranges from different cursors.
        ranges = sorted(ranges, key=lambda x: x[0])
        new_ranges = []

        prev = ranges[0]
        for cursor_range in ranges[1:]:
            if cursor_range[0] <= prev[1]:
                prev = (prev[0], cursor_range[1])
                continue
            new_ranges.append(prev)
            prev = cursor_range

        if prev not in new_ranges:
            new_ranges.append(prev)

        new_ranges = new_ranges[::-1]
        self.write_buffer.cursors = []

        # Remove all the ranges.
        for cursor_range in new_ranges:
            cursor = Cursor(parent=self.write_buffer)
            self.write_buffer.cursors.append(cursor)

            old_pos, new_pos = cursor_range
            middle = cursor.parent.text[old_pos - 1 : new_pos - 1]
            self.registers["copy"] = middle

            self.write_buffer.text = (
                self.write_buffer.text[: old_pos - 1]
                + self.write_buffer.text[new_pos - 1 :]
            )
            cursor.set_pointer(old_pos)


@dc.dataclass
class Cursor:
    """A cursor."""

    parent: TextBuffer = None
    pointer: int = 1
    vertical_movement_pointer: int = 0

    def __bool__(self):
        """Return whether or not the parent has text."""
        return bool(self.parent.text)

    def __lt__(self, other):
        """Return whether or not the pointer location is less than the other pointer location."""
        return self.pointer < other.pointer

    def reset_v_pointer(self):
        """Reset vertical movement."""
        self.vertical_movement_pointer = -1

    def move_pointer(self, steps, strict: bool = False) -> None:
        """Move the pointer forwards a number of steps or into the next match of some pattern."""
        if isinstance(steps, (str, re.Pattern)):
            match = re.match(steps, self.parent.text[self.pointer :])
            if match is None:
                assert not strict
            return self.move_pointer(len(match.groups(0)))

        assert isinstance(steps, int), (steps, type(steps))
        modified = self.pointer + steps
        if modified in range(1, len(self.parent.text) + 1):
            self.pointer = modified
        else:
            assert not strict
            self.pointer = max(1, min(len(self.parent.text), modified))

        dist_to_newline = self.parent.text[: self.pointer - 1][::-1].find("\n")

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

    def move_pointer_and_siblings(self, steps, strict: bool = False) -> None:
        """Move all pointers forwards a number of steps or into the next match of some pattern."""
        for cursor in self.parent.cursors:
            cursor.move_pointer(steps, strict=strict)

    def write(self, pos: int, chars: str, move_pointer: bool = True) -> None:
        """Write some text at a location in the string, moving relevant cursors if necessary."""
        self.parent.text = self.parent.text[:pos] + chars + self.parent.text[pos:]
        self.parent.write_callback(self.parent)

        if move_pointer:
            for c in self.parent.cursors:
                if c.pointer > pos:
                    c.pointer += len(chars)

    def delete(self, n: int) -> None:
        """Delete some text from the parent text."""
        self.parent.text = (
            self.parent.text[: max(0, self.pointer - n)]
            + self.parent.text[self.pointer :]
        )

    def write_with_callback(self, chars: str) -> None:
        """Write some text, calling the callback on newlines."""
        lines = chars.splitlines()
        n_lines = len(lines)

        for i, line in enumerate(lines):
            self.write(max(0, self.pointer - 1), line)

            # Avoid writing newline on the last line.
            if i == (n_lines - 1):
                continue

            if not self.parent.multiline:
                self.parent.newline_callback(self)
                continue

            self.write(max(0, self.pointer - 1), "\n")

    def send_text(self, key: int, name: str) -> None:
        """Handle sending text and similar modifications to the selected buffer."""
        mapping = {"Tab": TAB_CHAR}
        char = mapping.get(name, chr(key))

        # Handle Ctrl + V
        if name == "v":
            control_held_down = sc.HELD_DOWN["Control_R"] or sc.HELD_DOWN["Control_L"]

            if control_held_down:
                clipboard = utils.read_clipboard()
                self.write_with_callback(clipboard)
                return

        # Special characters.
        if name == "ISO_Left_Tab":
            line_index = utils.line_num2index(0, self.pointer, self.parent.text) + 1
            offset_to_newline = (
                self.pointer
                if line_index == 0
                else (self.parent.text[: self.pointer].rfind("\n") - self.pointer)
            )
            offset_to_newline = abs(offset_to_newline)

            line_index -= offset_to_newline
            line_index = max(0, line_index)
            line = self.parent.text[line_index:].splitlines()[0]
            original_line_length = len(line)

            if line.startswith(TAB_CHAR):
                line = line.removeprefix(TAB_CHAR)
            else:
                line = line.lstrip()

            # Actually make the move.
            self.move_pointer_and_siblings(len(line) - original_line_length)
            self.parent.text = (
                self.parent.text[:line_index:]
                + line
                + self.parent.text[line_index + original_line_length :]
            )

        elif name == "Left":
            self.handle_buffer_action(BufferAction.LEFT)
        elif name == "Right":
            self.handle_buffer_action(BufferAction.RIGHT)
        elif name == "Up":
            self.handle_buffer_action(BufferAction.UP)
        elif name == "Down":
            self.handle_buffer_action(BufferAction.DOWN)
        elif name == "Delete":
            self.delete(1)
        elif name == "BackSpace":
            self.move_pointer_and_siblings(-1)
            self.delete(1)

        elif char != "\x00":
            # Input the character.
            if char == "\r":
                if not self.parent.multiline:
                    self.parent.newline_callback(self)
                    return
                char = "\n"

            self.write(max(0, self.pointer - 1), char)

    def prev(self, char: str) -> None:
        """Go to the previous occurence of a regex."""
        return mov.prv_template(char)(self.pointer, self.parent.text)

    def next(self, char: str) -> None:
        """Go to the next occurence of a regex."""
        return (
            mov.nxt_template(
                char,
            )(self.pointer, self.parent.text)
            + 1
            - (
                (self.pointer - 1) in range(len(self.parent.text))
                and (self.parent.text[self.pointer - 1] == "\n")
            )
        )

    def current_line(self) -> int:
        """Get the current line of the code."""
        return self.parent.text[: self.pointer - 1].count("\n")

    def line_bounds(self, n=None) -> t.Tuple[int]:
        """Get the bounds of some line of the code, defaulting to the current line."""
        if n is None:
            n = self.current_line()

        lines = self.parent.text.splitlines(keepends=True)
        n = min(len(lines) - 1, n)

        old_pos = sum(map(len, lines[:n]))
        new_pos = old_pos + len(lines[n])

        return old_pos, new_pos

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
                    return
                pos = self.prev(r"[^\n]*\n")
            elif action == BufferAction.DOWN:
                # Go to the end of the buffer if on last line.
                if self.current_line() == self.parent.text.count("\n"):
                    self.reset_v_pointer()
                    self.set_pointer(int(1e20))
                    return

                pos = self.next(r"\n")
            else:
                raise NotImplementedError()

            line = self.parent.text[:pos].count("\n")

            # Move to the same location in the next line.
            lines = self.parent.text.splitlines(keepends=True)
            endl_0 = sum(map(len, lines[:line]))
            endl_1 = sum(map(len, lines[: line + 1]))
            pos = endl_0 + self.vertical_movement_pointer + 1

            # Make sure the position is on the right line.
            pos = min(endl_1, pos)
            self.set_pointer(pos)
        else:
            raise NotImplementedError()


@dc.dataclass
class TextWindow(sc.Container):
    """A window containing text."""

    active_buffer: ... = None

    focused_line: int = 0
    top_line: int = 0
    inner_spacing: int = 16

    font_size: int = 20
    char_ratio: float = 0.5  # The height/width ratio for Iosevka.
    text_v_spacing_ratio: int = 0.2

    cursor: sc.Rectangle = dc.field(default_factory=sc.Rectangle)
    body: sc.Container = dc.field(default_factory=sc.Container)
    body_buffer: TextBuffer = dc.field(default_factory=lambda: TextBuffer(DEFAULT_TEXT))

    @property
    def text_v_spacing(self):
        """Get the vertical spacing between different characters."""
        return self.font_size * self.text_v_spacing_ratio

    def _get_char_size(self, ctx) -> np.ndarray:
        """Get the size of a character in this font."""
        txt = sc.Text()
        ctx.select_font_face(txt.font_face, txt.font_slant, txt.font_weight)
        ctx.set_font_size(self.font_size)
        (_, _, width, height, _, _) = ctx.text_extents("█")

        return np.array([width, height])

    def _get_body_n_chars(self, ctx) -> np.ndarray:
        """Get a ndarray that contains the number of rows and columns available in the body."""
        width, height = self._get_char_size(ctx)
        return (self.body.size) / [width, height + self.text_v_spacing]

    def _get_pos(self, x: int, y: int, ctx) -> np.ndarray:
        """Get the position in pixels from the coordinates of the symbol."""
        return self.body.get_top_left() + [
            x * self._get_char_size(ctx)[0],
            (
                y * (self.text_v_spacing + self._get_char_size(ctx)[1])
                + self.text_v_spacing
                + 2 * self.font_size
            ),
        ]

    def _update_cursor(self, ctx, wide=True):
        """Update the cursor attribute to be drawn on screen."""
        if wide:
            self.cursor = sc.RoundedRectangle(
                size=[
                    self._get_char_size(ctx)[0] * 1.2,
                    self._get_char_size(ctx)[1] * 1.2,
                ],
                color=sc.utils.Color.ACCENT_1,
                radius=self._get_char_size(ctx)[0] * 0.3,
                anchor=sc.Anchor.LEFT,
            )
        else:
            self.cursor = sc.Rectangle(
                size=[
                    self._get_char_size(ctx)[0] * 0.2,
                    self._get_char_size(ctx)[1] * 1.5,
                ],
                color=sc.utils.Color.WHITE,
                anchor=sc.Anchor.LEFT,
            )

    def _update_body(self, ctx):
        """Update the body."""
        self._update_cursor(ctx, wide=self.current_mode.mode != INSERT)
        self._focus_selected_line(ctx)

        # Generate line number and line.
        def get_item(line, num=None):
            num = "·" if num is None else str(num + self.focused_line)
            num = num.rjust(5).ljust(LINE_NUM_CHARS)
            return num, line

        self.body_buffer.remove_duplicate_cursors()
        self.body.children = []
        n_drawn_lines = 0

        char_size = self._get_body_n_chars(ctx)
        char_size = list(map(int, char_size))
        char_size[0] -= 2
        assert all(i > 10 for i in char_size)

        cursor_queue = self.body_buffer.cursors.copy()

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
                sc.Text(
                    text=num,
                    color=sc.utils.Color.GREY_L1,
                    font_size=self.font_size,
                    position=self._get_pos(0, n_drawn_lines, ctx),
                    anchor=sc.Anchor.BOTTOM | sc.Anchor.LEFT,
                )
            )

            self.body.children.append(
                sc.ColoredText(
                    text=line.removesuffix("\n"),
                    default_state=sc.TextState(fore=sc.utils.Color.WHITE),
                    font_size=self.font_size,
                    position=self._get_pos(LINE_NUM_CHARS, n_drawn_lines, ctx),
                    anchor=sc.Anchor.BOTTOM | sc.Anchor.LEFT,
                )
            )

            i = 0

            while i < len(cursor_queue):
                cursor = cursor_queue[i]
                if n_chars < cursor.pointer:
                    i += 1
                    continue

                cursor_queue.pop(i)

                x_coord = cursor.pointer - n_chars + len(line) - 1
                y_coord = n_drawn_lines - 1
                self._draw_cursor_at(x_coord, y_coord, ctx)

            n_drawn_lines += 1

    def _draw_cursor_at(self, x, y, ctx):
        # Draw the pointer if it's on the right row of characters.
        self.cursor.position = self._get_pos(LINE_NUM_CHARS + x, y, ctx)

        if self.current_mode.mode != INSERT:
            self.cursor.color = self.body.children[-1].get_state(x).fore
            self.body.children[-1].set_state(
                x, x + 1, sc.TextState(fore=sc.utils.Color.BLACK)
            )
        selector = sc.RoundedRectangle(
            size=[
                self._get_char_size(ctx)[0] * 6,
                self._get_char_size(ctx)[1] * 1.3,
            ],
            color=sc.utils.Color.ACCENT_2,
            radius=self._get_char_size(ctx)[0] * 0.3,
            anchor=sc.Anchor.LEFT,
        )

        self.body.children[-2].color = sc.utils.Color.BLACK
        selector.position = self._get_pos(0, y, ctx)

        # Do the rendering.
        self.cursor.render(ctx)
        selector.render(ctx)

    def _focus_selected_line(self, ctx) -> None:
        """Make sure that the selected line is focused."""
        focused_pointer = self.body_buffer.cursors[-1].pointer
        lines = self.body_buffer.text.splitlines(keepends=True)

        acc = 0
        i = 0

        for i, line in enumerate(lines):
            acc += len(line)
            if acc >= focused_pointer:
                break

        self.top_line = i

        char_size = self._get_body_n_chars(ctx)
        lines_on_screen = int(char_size[1])

        top_line_to_show = i - (lines_on_screen // 2)
        top_line_to_show = max(0, top_line_to_show)
        self.focused_line = top_line_to_show


@dc.dataclass
class Editor(TextWindow):
    """The editor."""

    normal_handler: NormalModeHandler = dc.field(default_factory=NormalModeHandler)
    normal_buffer: TextBuffer = dc.field(
        default_factory=lambda: TextBuffer(text="", multiline=False)
    )
    shell_buffer: TextBuffer = dc.field(
        default_factory=lambda: TextBuffer(text="", multiline=False)
    )

    current_mode: ModeAndColor = MODES[NORMAL]
    top_info: sc.RoundedRectangle = dc.field(default_factory=sc.RoundedRectangle)
    mode_indicator: sc.RoundedRectangle = dc.field(default_factory=sc.RoundedRectangle)
    mode_buffer_container: sc.RoundedRectangle = dc.field(
        default_factory=sc.RoundedRectangle
    )

    def __post_init__(self):
        self.body_buffer.cursors = [
            Cursor(self.body_buffer, pointer=3),
            Cursor(self.body_buffer, pointer=10),
        ]

        self.active_buffer = self.body_buffer
        self.shell_buffer.newline_callback = lambda s: self._handle_shell_command(
            s.text
        )

        for i in (
            self.body,
            self.top_info,
            self.mode_indicator,
            self.mode_buffer_container,
        ):
            i.parent = self
            i.color = sc.utils.Color.ACCENT_0
            i.anchor = sc.Anchor.TOP | sc.Anchor.LEFT

        self.mode_buffer_container.color = sc.utils.Color.GREY_D1

        self.mode_indicator.children.append(
            sc.Text(
                text=self.current_mode.mode.name,
                font_size=self.font_size,
                font_weight=cairo.FONT_WEIGHT_BOLD,
            )
        )

        self.mode_buffer_container.children.append(sc.Text(font_size=self.font_size))

        for i in self.mode_indicator.children:
            i.parent = self.mode_indicator
        for i in self.mode_buffer_container.children:
            i.parent = self.mode_buffer_container

    def render(self, ctx) -> None:
        """Render the editor and its children."""
        # TODO: moldy code unless fixed

        spacing = self.inner_spacing
        top_left = self.get_top_left()

        # Place the top info box.
        self.top_info.position = top_left + spacing
        self.top_info.size = np.array(
            [
                self.size[0] - spacing * 2,
                self.font_size + self.text_v_spacing * 2,
            ]
        )

        # Place the bottom info box.
        self.mode_indicator.anchor = sc.Anchor.BOTTOM | sc.Anchor.LEFT
        self.mode_indicator.position = (
            top_left + [0, self.size[1]] + [spacing, -spacing]
        )
        self.mode_indicator.size = [
            (self.font_size / 2 * 6) + self.text_v_spacing * 2,
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

        self.mode_buffer_container.anchor = sc.Anchor.BOTTOM | sc.Anchor.LEFT

        # Set the text box positions.
        for child in self.mode_buffer_container.children:
            child.position = (
                child.parent.get_top_left()
                + np.array(child.parent.size) * [0, 0.5]
                + self.text_v_spacing
            )
            child.anchor = sc.Anchor.TOP | sc.Anchor.LEFT
            child.color = sc.utils.Color.WHITE
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
                + [0, self.text_v_spacing]
            )
            child.anchor = sc.Anchor.TOP
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
        control_held_down = sc.HELD_DOWN["Control_R"] or sc.HELD_DOWN["Control_L"]

        # Switch back to Normal mode on escape.
        if name == "Escape":
            if current != NORMAL:
                self.switch_mode(NORMAL)
                return
            self.body_buffer.remove_all_but_one_cursor()
            return

        if control_held_down:
            if name == "plus":
                self.font_size += 3
                return
            if name == "minus":
                self.font_size -= 3
                self.font_size = max(5, self.font_size)
                return

        if current == NORMAL:
            self._handle_normal_mode(key, name)
            return
        if current in {SHELL, INSERT}:
            self.active_buffer.send_text(key, name)
            return
        raise NotImplementedError()

    def _handle_normal_mode(self, key: int, name: str):
        """Handle normal mode."""
        self.normal_buffer.send_text(key, name)
        self.normal_handler.read_buffer = self.normal_buffer
        self.normal_handler.write_buffer = self.active_buffer

        if name == "Escape":
            self.normal_buffer.clear()
        elif name == "period":
            self.switch_mode(SHELL)
            self.normal_buffer.clear()
        elif name == "i":
            self.switch_mode(INSERT)
            self.normal_buffer.clear()

        return self.normal_handler.parse_command()

    def _handle_shell_command(self, command: str) -> None:
        """Handle a shell command when it's entered."""
        command = command.strip()

        if command == "q":
            return self.kill()

        self.shell_buffer.text = ""
        return None
