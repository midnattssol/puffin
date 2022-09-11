#!/usr/bin/env python3.10
"""Utilities."""
import pandas as pd


def read_clipboard() -> str:
    """Read the current clipboard contents."""
    dataframe = pd.read_clipboard()
    return dataframe.columns[0]


def line_num2index(num: int, pos: int, buf: str) -> int:
    """Get the line number from an index."""

    def inner(pos, left):
        if left == 0:
            return pos, left
        if num < 0:
            _buf = buf[: max(0, pos - 2)]
            idx = _buf.rfind("\n")
            if idx == -1:
                return (-1, None)
        if num > 0:
            _buf = buf[pos - 1 :]
            idx = _buf.find("\n")
            if idx == -1:
                return (len(buf) - 1, None)
            idx += pos - 1

        return inner(idx, left - 1)

    left = abs(num)
    result, _ = inner(pos, left)
    return result
