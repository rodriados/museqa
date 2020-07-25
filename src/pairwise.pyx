#!/usr/bin/env python
# Multiple Sequence Alignment pairwise wrapper file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2020 Rodrigo Siqueira
from libc.stdint cimport *
from libcpp.string cimport string
from libcpp.vector cimport vector
from cartesian cimport c_cartesian2
from database cimport Database
from encoder cimport c_encode
from pairwise cimport *

# Exposes the module's resulting distance matrix.
# @since 0.1.1
cdef class DistanceMatrix:
    # Accesses a value on the distance matrix.
    # @param offset The requested matrix position to access.
    # @return The score of given position.
    def __getitem__(self, tuple offset):
        cdef uint32_t x = int(offset[0])
        cdef uint32_t y = int(offset[1])
        cdef c_cartesian2[size_t] position = c_cartesian2[size_t](x, y)

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
    def __cinit__(self, str name = 'default'):
        cdef string tablename = name.encode('ascii')
        self.thisptr = c_scoring_table.make(tablename)

    # Accesses a value on the scoring table.
    # @param offset The requested table position offset.
    # @return The score of given tuple index.
    def __getitem__(self, tuple offset):
        cdef uint8_t x = c_encode(ord(offset[0])) if isinstance(offset[0], str) else int(offset[0])
        cdef uint8_t y = c_encode(ord(offset[1])) if isinstance(offset[1], str) else int(offset[1])
        cdef c_cartesian2[intptr_t] point = c_cartesian2[intptr_t](x, y)

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
    algoname = kwargs.pop('algorithm', 'default').encode('ascii')

    cdef ScoringTable s_table = table if type(table) is not str else ScoringTable(table)
    return DistanceMatrix.wrap(c_run(db.thisptr, s_table.thisptr, algoname))
