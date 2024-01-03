import operator
from typing import Callable, List, Tuple

import pytest

from timebasedcv.utils._funcs import pairwise, pairwise_comparison


def test_pairwise(sample_list: List[int], sample_pairs: List[Tuple[int, int]]):
    """Tests pairwise function."""
    assert list(pairwise(sample_list)) == sample_pairs


@pytest.mark.parametrize(
    "_operation, expected",
    [
        (operator.lt, [True, True, True, True]),
        (operator.gt, [False, False, False, False]),
        (operator.eq, [False, False, False, False]),
    ],
)
def test_pairwise_comparison(sample_list: List[int], _operation: Callable[[int, int], bool], expected: List[bool]):
    """Tests pairwise_comparison function."""
    assert list(pairwise_comparison(sample_list, _operation)) == expected
