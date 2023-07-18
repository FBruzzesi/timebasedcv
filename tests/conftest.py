from typing import List, Tuple

import pytest


@pytest.fixture
def sample_list() -> List[int]:
    """Returns a sample list."""
    return [1, 2, 3, 4, 5]


@pytest.fixture
def sample_pairs() -> List[Tuple[int, int]]:
    """Returns a sample list of pairs."""
    return [(1, 2), (2, 3), (3, 4), (4, 5)]
