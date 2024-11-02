from __future__ import annotations

import operator
from typing import Callable
from typing import List
from typing import Tuple

import pytest

from timebasedcv.utils._funcs import pairwise
from timebasedcv.utils._funcs import pairwise_comparison


def test_pairwise(sample_list: List[int], sample_pairs: List[Tuple[int, int]]):
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
def test_pairwise_comparison(sample_list: List[int], op: Callable[[int, int], bool], expected: List[bool]):
    """Tests pairwise_comparison function."""
    assert list(pairwise_comparison(sample_list, op)) == expected
