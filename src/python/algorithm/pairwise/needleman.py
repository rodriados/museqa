#!/usr/bin/env python
# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file A reference implementation for the Needleman-Wunsch algorithm.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2021-present Rodrigo Siqueira
from museqa.pairwise import ScoringTable, DistanceMatrix
from museqa.database import Database
from museqa.sequence import Sequence

from .utils import generator

__all__ = ["needleman"]

# Executes the sequential Needleman-Wunsch algorithm.
# @param db The sequences available for alignment.
# @param table The scoring table to use.
# @param gap The gap opening penalty.
# @param extend The gap extending penalty.
# @return The distance matrix of score between sequence pairs.
def needleman(
    db: Database
,   *
,   table: ScoringTable = None
,   gap: float = None
,   extend: float = None
) -> DistanceMatrix:

    # Checking validating and initializing the parameters given to the function.
    table = table if table is not None else ScoringTable()
    #extend = extend if extend is not None else table.penalty
    #gap = gap if gap is not None else 0.0

    gap = gap if gap is not None else table.penalty
    
    assert type(table) is ScoringTable

    # Sequentially aligns two sequences using Needleman-Wunsch algorithm.
    # @param one The first sequence to align.
    # @param two The second sequence to align.
    # @return The alignment score.
    def align(one: Sequence, two: Sequence) -> float:
        if one.length <= two.length:
            one, two = two, one

        # Filling 0-th line with penalties. This is the only initialization needed
        # for sequential algorithm.
        line = [i * -gap for i in range(two.length + 1)]

        for i in range(one.length):
            # If the current line is at sequence end, then we can already finish
            # the algorithm, as no changes are expected to occur after the end 
            # of sequence.
            if one[i] == Sequence.padding:
                break

            # Initialize the 0-th column values. It will always be initialized with
            # penalties, in the same manner as the 0-th line.
            done = line[0]
            line[0] = (i + 1) * -gap

            # Iterate over the second sequence, calculating the best alignment possible
            # for each of its characters.
            for j in range(1, two.length + 1):
                value = line[j - 1]

                if two[j - 1] != Sequence.padding:
                    insertd = value - gap
                    removed = line[j] - gap
                    matched = done + table[one[i], two[j - 1]]
                    value = max(insertd, removed, matched)

                done = line[j]
                line[j] = value

        return line[-1]

    distances = [align(*pair) for pair in generator(db)]
    return DistanceMatrix(distances, db.count)
