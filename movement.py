#!/usr/bin/env python3.10
"""Handles movement in strings."""
import functools as ft

import regex as re


# ===| Functions |===


def nxt_template(regex: str, include_span: bool = True) -> callable:
    def inner(pos: int, string: str) -> int:
        match = re.search(regex, string[pos:])
        if match is None:
            return -1
        return match.span()[include_span] + pos - 1

    return inner


def prv_template(regex: str, include_span: bool = True) -> callable:
    def inner(pos: int, string: str) -> int:
        match = re.finditer(regex, string)
        last_item = None

        for item in match:
            num = item.span()[1]
            if num >= pos:
                break
            last_item = item

        if last_item is not None:
            return last_item.span()[not include_span]

        return 1

    return inner


# def jump_enclosing_paren_template(p0: str, p1: str, rev=False):
#     def inner(pos: int, string: str):
#         i = 1 if rev else -1
#         depth = 0
#         pos = 0
#
#         while True:
#
#
#
#         return i
#
#     return inner
# jump_enclosing = jump_enclosing_paren_template("([{", "}])")

nxt_word_start = nxt_template(r"\s+")
nxt_word_end = nxt_template(r"\w+")
nxt_line_start = nxt_template(r"\n")
nxt_line_end = nxt_template(r"\n", include_span=False)

prv_word_start = prv_template(r"\s+")
prv_word_end = prv_template(r"\w+")
prv_line_start = prv_template(r"\n")
prv_line_end = prv_template(r"\n", include_span=False)


if __name__ == "__main__":
    long = """hello world this is a
    multiline string!"""

    assert nxt_word_start(0, "hello world") == len("hello ")
    assert nxt_word_end(0, "hello world") == len("hello")
    assert nxt_word_end(5, "hello world") == len("hello world")

    assert nxt_line_start(0, long) == len(long.splitlines()[0]) + 1
    assert nxt_line_end(0, long) == len(long.splitlines()[0])

    assert prv_line_end(len(long), long) == len(long.splitlines()[0])
