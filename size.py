#!/usr/bin/env python3.10
""""""
import dataclasses as dc
import utils
import typing as t
import time
import interpolate as ip

# ===| Sizes |===
@dc.dataclass
class Size:
    n: float


@dc.dataclass
class Absolute(Size):
    def get(self, parent_size=None, parent=None):
        return self.n


@dc.dataclass
class Relative(Size):
    def get(self, parent_size=None, parent=None):
        return self.n * parent_size


@dc.dataclass
class Percentage(Size):
    def get(self, parent_size=None, parent=None):
        return self.n * parent_size * 100


@dc.dataclass
class Offset(Size):
    def get(self, parent_size=None, parent=None):
        return self.n + parent_size


@dc.dataclass
class ParentFunction(Size):
    def get(self, parent_size=None, parent=None):
        return self.n(parent_size, parent)


@dc.dataclass
class TimeFunction(Size):
    start_time: float = dc.field(default_factory=time.time)

    def get(self, parent_size=None, parent=None):
        dt = time.time() - self.start_time
        return self.n(dt)


@dc.dataclass
class Easing(Size):
    target: t.Any
    period: float
    start_time: float = dc.field(default_factory=time.time)
    easing: t.Optional[callable] = None
    _is_expired: bool = False
    interpolator: callable = ip.lerp
    clamp: bool = True

    @property
    def origin(self):
        return self.n

    @property
    def is_expired(self):
        """Get whether or not the process has expired."""
        self._is_expired = self._is_expired or time.time() >= (
            self.start_time + self.period
        )
        return self._is_expired

    def get_frac(self):
        frac = (time.time() - self.start_time) / self.period
        if self.clamp:
            frac = min(max(frac, 0), 1)
        return frac

    def get(self, parent_size=None, parent=None):
        """Get the current value."""
        frac = self.get_frac()

        if self.easing is not None:
            frac = self.easing(frac)

        return self.interpolator(self.origin, self.target, frac)
