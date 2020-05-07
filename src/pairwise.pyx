#!/usr/bin/env python
# cython: language_level = 3
# Multiple Sequence Alignment pairwise wrapper file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2019 Rodrigo Siqueira
from libc.stdint cimport *
from libcpp.string cimport string
from libcpp.vector cimport vector
from database cimport Database
from encoder cimport c_encode
from pairwise cimport *

# Manages all data and execution of the pairwise module.
# @since 0.1.1
cdef class Pairwise:
    # Gets an item from the module's result.
    # @param offset The requested element offset.
    # @return The element in requested offset.
    def __getitem__(self, tuple offset):
        cdef c_cartesian2 point = c_cartesian2(int(offset[0]), int(offset[1]))
        cdef c_score score = self.thisptr.at(point)
        return score

    # Aligns every sequence in given database pairwise, thus calculating a similarity
    # score for every different permutation of sequence pairs.
    # @param db The database to be processed.
    # @param table The chosen scoring table.
    # @param algorithm The pairwise algorithm to use.
    # @return The processed pairwise instance.
    @staticmethod
    def run(Database db, **kwargs):
        cdef string algorithm = kwargs.pop('algorithm', 'default').encode('ascii')
        cdef string table = kwargs.pop('table', 'default').encode('ascii')

        return Pairwise.wrap(c_pairwise.run(configure(db.thisptr, algorithm, table)))

    # Wraps an existing pairwise instance.
    # @param target The pairwise to be wrapped.
    # @return The new wrapper instance.
    @staticmethod
    cdef Pairwise wrap(c_pairwise& target):
        instance = <Pairwise>Pairwise.__new__(Pairwise)
        instance.thisptr = target
        return instance

    # Informs the number of processed pairs or to process.
    # @return The number of pairs this instance shall process.
    @property
    def count(self):
        return self.thisptr.count()

# Exposes a scoring table to Python world.
# @since 0.1.1
cdef class ScoringTable:
    # Instantiates a new scoring table instance.
    # @param name The selected scoring table name.
    def __cinit__(self, str name = 'default'):
        cdef string tablename = name.encode('ascii')
        self.thisptr = c_scoring_table.make(tablename)

    # Accesses a value in the scoring table.
    # @param offset The requested table position offset.
    # @return The score of given tuple index.
    def __getitem__(self, tuple offset):
        cdef uint8_t x = c_encode(ord(offset[0])) if isinstance(offset[0], str) else int(offset[0])
        cdef uint8_t y = c_encode(ord(offset[1])) if isinstance(offset[1], str) else int(offset[1])
        cdef c_cartesian2 point = c_cartesian2(x, y)

        if x >= 25 or y >= 25:
            raise RuntimeError("scoring table offset out of range")

        return self.thisptr.at(point)

    # Gets the list of every scoring table available.
    # @return The list of all scoring tables available.
    @staticmethod
    def list():
        cdef vector[string] result = c_scoring_table.list()
        return [elem.decode() for elem in result]

    # Wraps an existing scoring table instance.
    # @param target The scoring table to be wrapped.
    # @return The new wrapper instance.
    @staticmethod
    cdef ScoringTable wrap(c_scoring_table& target):
        instance = <ScoringTable>ScoringTable.__new__(ScoringTable)
        instance.thisptr = target
        return instance

    # Gives access to the table's penalty value.
    # @return The table's penalty value.
    @property
    def penalty(self):
        return self.thisptr.penalty()

# Lists all available algorithms to this module.
# @return The list of all algorithms available.
def algorithms():
    cdef vector[string] result = c_algorithm.list()
    return [elem.decode() for elem in result]

# Aligns every sequence in given database pairwise, thus calculating a similarity
# score for every different permutation of sequence pairs.
# @param db The database to be processed.
# @param table The chosen scoring table.
# @param algorithm The pairwise algorithm to use.
# @return The pairwise module instance.
def run(Database db, **kwargs):
    return Pairwise(db, **kwargs)
