#!/usr/bin/env python3.10
"""Utilities."""
import pandas as pd


def read_clipboard() -> str:
    """Read the current clipboard contents."""
    df = pd.read_clipboard()
    return df.columns[0]
