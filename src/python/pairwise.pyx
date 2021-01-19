#!/usr/bin/env python
# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file Implementation for the pairwise module wrapper.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-present Rodrigo Siqueira
from libc.stdint cimport *
from libcpp.string cimport string
from libcpp.vector cimport vector
from database cimport Database
from encoder cimport c_encode
from point cimport c_point2
from pairwise cimport *

# Exposes the module's resulting distance matrix.
# @since 0.1.1
cdef class DistanceMatrix:
    # Instantiates a new distance matrix from a list of pairwise sequence scores.
    # @param score The list of pair distances.
    # @param count The total number of sequences represented.
    def __cinit__(self, list score = [], int count = 0):
        cdef vector[c_score] buf = score
        self.thisptr = c_dist_matrix(buf, count)

    # Accesses a value on the distance matrix.
    # @param offset The requested matrix position to access.
    # @return The score of given position.
    def __getitem__(self, tuple offset):
        cdef uint32_t x = int(offset[0])
        cdef uint32_t y = int(offset[1])
        cdef c_point2[size_t] position = c_point2[size_t](x, y)

        return self.thisptr.at(position)

    # Informs the matrix's dimension.
    # @return The distance matrix's dimension.
    @property
    def count(self):
        return self.thisptr.count()

# Exposes the module's scoring table object.
# @since 0.1.1
cdef class ScoringTable:
    # Instantiates a new scoring table instance.
    # @param name The selected scoring table name.
    def __cinit__(self, str name = None):
        name = name if name is not None else 'default'
        cdef string tablename = name.encode('ascii')
        self.thisptr = c_scoring_table.make(tablename)

    # Accesses a value on the scoring table.
    # @param offset The requested table position offset.
    # @return The score of given tuple index.
    def __getitem__(self, tuple offset):
        cdef uint8_t x = c_encode(ord(offset[0])) if isinstance(offset[0], str) else int(offset[0])
        cdef uint8_t y = c_encode(ord(offset[1])) if isinstance(offset[1], str) else int(offset[1])
        cdef c_point2[size_t] point = c_point2[size_t](x, y)

        if x >= 25 or y >= 25:
            raise RuntimeError("scoring table offset out of range")

        return self.thisptr.at(point)

    # Gets the list of every scoring table available.
    # @return The list of all scoring tables available.
    @staticmethod
    def list():
        cdef vector[string] result = c_scoring_table.list()
        return [elem.decode() for elem in result]

    # Gives access to the table's penalty value.
    # @return The table's penalty value.
    @property
    def penalty(self):
        return self.thisptr.penalty()

# Lists all available algorithms to this module.
# @return The list of all algorithms available.
def list():
    cdef vector[string] result = c_algorithm.list()
    return [elem.decode() for elem in result]

# Aligns every sequence in given database pairwise, thus calculating a similarity
# score for every different permutation of sequence pairs.
# @param db The database to be processed.
# @param table The chosen scoring table.
# @param algorithm The pairwise algorithm to use.
# @return The resulting distance matrix.
def run(Database db, **kwargs):
    table = kwargs.pop('table', ScoringTable())
    algorithm = kwargs.pop('algorithm', 'default')

    cdef ScoringTable s_table = table if type(table) is ScoringTable else ScoringTable(table)

    if type(algorithm) is str:
        return DistanceMatrix.wrap(c_run(db.thisptr, s_table.thisptr, algorithm.encode('ascii')))

    cdef DistanceMatrix matrix = algorithm(db, table = s_table)
    return matrix
