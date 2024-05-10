from itertools import tee
from typing import Callable, Iterable, Tuple, TypeVar

T = TypeVar("T")


def pairwise(iterable: Iterable[T]) -> Iterable[Tuple[T, T]]:
    """Returns an iterator that yields pairs of consecutive elements from the given iterable.

    s -> (s0, s1), (s1, s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


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
    return (comparison_op(a, b) for a, b in pairwise(iterable))
