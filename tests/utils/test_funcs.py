from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import pytest

from timebasedcv.utils._funcs import pairwise
from timebasedcv.utils._funcs import pairwise_comparison

if TYPE_CHECKING:
    from collections.abc import Callable


def test_pairwise(sample_list: list[int], sample_pairs: list[tuple[int, int]]):
    """Tests pairwise function."""
    assert list(pairwise(sample_list)) == sample_pairs


@pytest.mark.parametrize(
    "op, expected",
    [
        (operator.lt, [True, True, True, True]),
        (operator.gt, [False, False, False, False]),
        (operator.eq, [False, False, False, False]),
    ],
)
def test_pairwise_comparison(sample_list: list[int], op: Callable[[int, int], bool], expected: list[bool]):
    """Tests pairwise_comparison function."""
    assert list(pairwise_comparison(sample_list, op)) == expected
