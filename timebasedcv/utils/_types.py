from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Protocol, Tuple, TypeVar, Union

import pandas as pd

DateTimeLike = TypeVar("DateTimeLike", datetime, date, pd.Timestamp)


FrequencyUnit = Literal[
    "days", "seconds", "microseconds", "milliseconds", "minutes", "hours", "weeks"
]

WindowType = Literal["rolling", "expanding"]

T = TypeVar("T")


class SeriesLike(Protocol[T]):
    """
    SeriesLike protocol for type hinting purposes.

    This protocol is used to indicate that the class should supports:

    - comparison operators: `__lt__`, `__gt__`, `__le__`, `__ge__`
    - logical operators: `__and__`
    - `.min()` and `.max()` methods
    - `.shape` attribute
    """

    def min(self) -> T:
        """Support for `.min()` method."""
        ...

    def max(self) -> T:
        """Support for `.max()` method."""
        ...

    @property
    def shape(self) -> Tuple[int]:
        ...

    def __lt__(self, other: Union[T, SeriesLike[T]]) -> SeriesLike[bool]:
        ...

    def __gt__(self, other: Union[T, SeriesLike[T]]) -> SeriesLike[bool]:
        ...

    def __le__(self, other: Union[T, SeriesLike[T]]) -> SeriesLike[bool]:
        ...

    def __ge__(self, other: Union[T, SeriesLike[T]]) -> SeriesLike[bool]:
        ...

    def __and__(self: SeriesLike[bool], other: SeriesLike[bool]) -> SeriesLike[bool]:
        ...


T_co = TypeVar("T_co", covariant=True)


class TensorLike(Protocol[T_co]):
    """
    TensorLike protocol for type hinting purposes.

    This protocol is used to indicate that the class should supports:

    - `.shape` attribute
    """

    @property
    def shape(self) -> Tuple[int, ...]:
        ...
