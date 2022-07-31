#!/usr/bin/env python3.10
import dataclasses as dc
import subprocess as sp
import typing as t


class Poller:
    command: str = "echo 'hello world'"
    timespan: float = 1
    formatter = lambda x: x

    last_timestamp = None
    raw = None
    process = None

    def get(self):
        """Get the value."""
        self.update()
        if self.raw is None:
            return None
        return self.formatter(self.raw)

    def update(self):
        """Run the command again in case enough time has passed."""

        should_update = (
            self.last_timestamp is None
            or time.time() - self.last_timestamp > self.timespan
        )

        if should_update:
            # Call functions normally.
            if callable(self.command):
                self.raw = self.command()
                return None

            # Start a new process.
            self.last_timestamp = time.time()
            self.process = sp.Popen(
                self.command, shell=True, stdout=sp.PIPE, stderr=sp.DEVNULL
            )

            if self.process is not None and self.process.poll() is not None:
                self.raw = self.process.stdout.read().strip()
                self.process = None
