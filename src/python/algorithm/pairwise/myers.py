#!/usr/bin/env python
# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file A reference implementation for the Myers-Miller algorithm.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2021-present Rodrigo Siqueira
from museqa.pairwise import ScoringTable, DistanceMatrix
from museqa.database import Database
from museqa.sequence import Sequence
from typing import Optional

from .utils import generator

__all__ = ['myers']

# Executes the sequential Myers-Miller algorithm.
# @param db The sequences available for alignment.
# @param table The scoring table to use.
# @param gap The gap opening penalty.
# @param extend The gap extending penalty.
# @return The distance matrix of score between sequence pairs.
def myers(
    db: Database
,   *
,   table: ScoringTable = None
,   gap: float = None
,   extend: float = None
) -> DistanceMatrix:

    # Checking validating and initializing the parameters given to the function.
    table = table if table is not None else ScoringTable()
    extend = extend if extend is not None else table.penalty
    gap = gap if gap is not None else 0.0
    
    assert type(table) is ScoringTable

    # Sequentially aligns two sequences using Myers-Miller algorithm.
    # @param one The first sequence to align.
    # @param two The second sequence to align.
    # @return The alignment score.
    def align(one: Sequence, two: Sequence) -> float:
        # Filling 0-th line with penalties. This is the only initialization needed
        # for sequential algorithm.
        line = [0] + [-(gap + extend * i) for i in range(1, two.length + 1)]

    distances = [align(*pair) for pair in generator(db)]
    return DistanceMatrix(distances, db.count)
