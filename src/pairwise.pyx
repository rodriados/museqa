#!/usr/bin/env python
# cython: language_level = 2
# Multiple Sequence Alignment pairwise wrapper file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2019 Rodrigo Siqueira
from cython.operator cimport dereference
from libcpp.string cimport string
from database cimport Database
from sequence cimport cencode
from pairwise cimport *

# Manages all data and execution of the pairwise module.
# @since 0.1.1
cdef class Pairwise:
    
    # Gets an item from the module's result.
    # @param offset The requested element offset.
    # @return The element in requested offset.
    def __getitem__(self, int offset):
        return self.cRef[offset]

    # Aligns every sequence in given database pairwise, thus calculating a similarity
    # score for every different permutation of sequence pairs.
    # @param db The database to be processed.
    # @param table The chosen scoring table.
    # @param algorithm The pairwise algorithm to use.
    def run(self, Database db, **kwargs):
        cdef string algorithm = kwargs.pop("algorithm", "needleman")
        cdef string table = kwargs.pop("table", "blosum62")

        self.cRef.run(db.cRef, algorithm, table)

    @property
    # Informs the number of processed pairs or to process.
    # @return The number of pairs this instance shall process.
    def count(self):
        return self.cRef.getCount()

# Exposes a scoring table to Python world.
# @since 0.1.1
cdef class ScoringTable:

    # Instantiates a new scoring table instance.
    # @param name The selected scoring table name.
    def __cinit__(self, str name):
        self.cPtr = cgetTable(name)

    # Accesses a value in the scoring table.
    # @param index The tuple of requested index.
    # @return The score of given tuple index.
    def __getitem__(self, tuple index):
        cdef uint8_t x, y

        x = cencode(ord(index[0][0])) if isinstance(index[0], str) else index[0]
        y = cencode(ord(index[1][0])) if isinstance(index[1], str) else index[1]

        if x >= 25 or y >= 25:
            raise RuntimeError("scoring table index out of range")

        return dereference(self.cPtr)[x][y]

# Aligns every sequence in given database pairwise, thus calculating a similarity
# score for every different permutation of sequence pairs.
# @param db The database to be processed.
# @param table The chosen scoring table.
# @param algorithm The pairwise algorithm to use.
# @return The pairwise module instance.
def run(Database db, **kwargs):
    pairwise = Pairwise()
    pairwise.run(db, **kwargs)
    return pairwise

# Gets a scoring table instance.
# @param table The name of requested scoring table.
# @return The scoring table instance.
def getTable(str table):
    if table not in getTableList():
        raise RuntimeError("could not find scoring table: " + table)

    return ScoringTable(table)

# Gets the list of every scoring table available.
# @return The list of all scoring tables available.
def getTableList():
    return cgetTableList()
