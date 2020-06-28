#!/usr/bin/env python
# Multiple Sequence Alignment pairwise export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2020 Rodrigo Siqueira
from libc.stdint cimport *
from libcpp.string cimport string
from libcpp.vector cimport vector
from cartesian cimport c_cartesian2
from database cimport c_database

cdef extern from "pairwise/pairwise.cuh" namespace "msa::pairwise":
    # The score of a sequence pair alignment.
    # @since 0.1.1
    ctypedef float c_score "msa::pairwise::score"

    # Represents a pairwise distance matrix. At last, this object represents
    # the pairwise module's execution's final result.
    # @since 0.1.1
    cdef cppclass c_dist_matrix "msa::pairwise::distance_matrix":
        ctypedef c_score element_type

        c_dist_matrix()
        c_dist_matrix(c_dist_matrix&)

        c_dist_matrix& operator=(c_dist_matrix&)
        element_type& at "operator[]" (c_cartesian2[size_t]&) except +RuntimeError

        c_cartesian2[size_t] dimension()

    # The aminoacid substitution tables. These tables are stored contiguously
    # in memory, in order to facilitate accessing its elements.
    # @since 0.1.1
    cdef cppclass c_scoring_table "msa::pairwise::scoring_table":
        ctypedef c_score element_type

        c_scoring_table()
        c_scoring_table(c_scoring_table&)

        c_scoring_table& operator=(c_scoring_table&) except +RuntimeError
        element_type at "operator[]" (c_cartesian2[intptr_t]&)

        element_type penalty()

        @staticmethod
        c_scoring_table make(string&) except +RuntimeError

        @staticmethod
        vector[string]& list()

    # Represents a pairwise module algorithm.
    # @since 0.1.1
    cdef cppclass c_algorithm "msa::pairwise::algorithm":
        @staticmethod
        vector[string]& list()

    # Runs the pairwise module with given parameters.
    cdef c_dist_matrix c_run "msa::pairwise::run" (c_database&, c_scoring_table&, string&) except +RuntimeError

# Exposes the module's resulting distance matrix.
# @since 0.1.1
cdef class DistanceMatrix:
    cdef c_dist_matrix thisptr

    # Sets the underlying C++ object to the given target instance.
    # @param target The target object to use as underlying instance.
    cdef inline void c_set(self, const c_dist_matrix& target):
        self.thisptr = target

    # Retrieves the instance's underlying C++ object.
    # @return The underlying C++ object instance.
    cdef inline c_dist_matrix c_get(self):
        return self.thisptr

    # Wraps an existing distance matrix instance.
    # @param target The distance matrix to be wrapped.
    # @return The new wrapper instance.
    @staticmethod
    cdef inline DistanceMatrix wrap(const c_dist_matrix& target):
        cdef DistanceMatrix wrapper = DistanceMatrix.__new__(DistanceMatrix)
        wrapper.c_set(target)
        return wrapper

# Exposes the module's scoring table object.
# @since 0.1.1
cdef class ScoringTable:
    cdef c_scoring_table thisptr

    # Sets the underlying C++ object to the given target instance.
    # @param target The target object to use as underlying instance.
    cdef inline void c_set(self, const c_scoring_table& target):
        self.thisptr = target

    # Retrieves the instance's underlying C++ object.
    # @return The underlying C++ object instance.
    cdef inline c_scoring_table c_get(self):
        return self.thisptr

    # Wraps an existing scoring table instance.
    # @param target The scoring table to be wrapped.
    # @return The new wrapper instance.
    @staticmethod
    cdef inline ScoringTable wrap(const c_scoring_table& target):
        cdef ScoringTable wrapper = ScoringTable.__new__(ScoringTable)
        wrapper.c_set(target)
        return wrapper
