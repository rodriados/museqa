#!/usr/bin/env python
# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file The pairwise algorithm's module utilities package.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2021-present Rodrigo Siqueira
from museqa.database import Database
from typing import Iterable

__all__ = ["generator"]

# Generates the list of sequence pairs from a database.
# @yield The pairs of sequences to be aligned.
def generator(db: Database) -> Iterable:
    yield from [
        (db[i].contents, db[j].contents)
            for i in range(1, db.count)
            for j in range(i)
    ]
