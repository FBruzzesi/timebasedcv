from __future__ import annotations

from datetime import date
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Literal
from typing import Protocol
from typing import TypeVar
from typing import Union

if TYPE_CHECKING:
    import pandas as pd
    from typing_extensions import Self  # pragma: no cover
    from typing_extensions import TypeAlias  # pragma: no cover

DateTimeLike = TypeVar("DateTimeLike", datetime, date, "pd.Timestamp")
NullableDatetime = Union[DateTimeLike, None]

FrequencyUnit: TypeAlias = Literal[
    "days", "seconds", "microseconds", "milliseconds", "minutes", "hours", "weeks", "months", "years"
]
WindowType: TypeAlias = Literal["rolling", "expanding"]
ModeType: TypeAlias = Literal["forward", "backward"]

T = TypeVar("T")


class SeriesLike(Protocol[T]):
    """SeriesLike protocol for type hinting purposes.

    This protocol is used to indicate that the class should supports:

    - comparison operators: `__lt__`, `__gt__`, `__le__`, `__ge__`
    - logical operators: `__and__`
    - `.min()` and `.max()` methods
    - `.shape` attribute
    """

    def min(self: Self) -> T: ...

    def max(self: Self) -> T: ...

    @property
    def shape(self: Self) -> tuple[int]: ...

    def __lt__(self: Self, other: T | SeriesLike[T]) -> SeriesLike[bool]: ...

    def __gt__(self: Self, other: T | SeriesLike[T]) -> SeriesLike[bool]: ...

    def __le__(self: Self, other: T | SeriesLike[T]) -> SeriesLike[bool]: ...

    def __ge__(self: Self, other: T | SeriesLike[T]) -> SeriesLike[bool]: ...

    def __and__(self: SeriesLike[bool], other: SeriesLike[bool]) -> SeriesLike[bool]: ...

    def __len__(self: Self) -> int: ...


T_co = TypeVar("T_co", covariant=True)


class TensorLike(Protocol[T_co]):
    """TensorLike protocol for type hinting purposes.

    This protocol is used to indicate that the class should supports:

    - `.shape` attribute
    """

    @property
    def shape(self: Self) -> tuple[int, ...]: ...

    def __getitem__(self: Self, i: slice | SeriesLike[bool] | SeriesLike[int]) -> Self: ...

    def __len__(self: Self) -> int: ...
