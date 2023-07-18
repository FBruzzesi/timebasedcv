from __future__ import annotations

from typing import Literal, Protocol, Tuple, TypeVar, Union

FrequencyUnit = Literal[
    "days", "seconds", "microseconds", "milliseconds", "minutes", "hours", "weeks"
]

WindowType = Literal["rolling", "expanding"]

T = TypeVar("T")


class SeriesLike(Protocol[T]):
    """
    SeriesLike protocol for type hinting purposes.
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


T_cov = TypeVar("T_cov", covariant=True)


class TensorLike(Protocol[T_cov]):
    """
    TensorLike protocol for type hinting purposes.
    """

    @property
    def shape(self) -> Tuple[int, ...]:
        ...
