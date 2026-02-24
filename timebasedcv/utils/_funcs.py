from __future__ import annotations

from itertools import pairwise
from itertools import starmap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from typing import TypeVar

    T = TypeVar("T")

__all__ = ("pairwise", "pairwise_comparison")


def pairwise_comparison(iterable: Iterable[T], comparison_op: Callable[[T, T], bool]) -> Iterable[bool]:
    """Apply pairwise comparison on consecutive elements.

    Returns an iterator that yields the result of applying the given comparison operator to pairs of consecutive
    elements from the given iterable.

    s -> (s0, s1), (s1, s2), ... -> comparison_op(s0, s1), comparison_op(s1, s2), ...

    Arguments:
        iterable: The iterable to iterate over.
        comparison_op: The comparison operator to apply to the pairs of consecutive
            elements.

    Returns:
        An iterator that yields the result of applying the given comparison operator.
    """
    return starmap(comparison_op, pairwise(iterable))
