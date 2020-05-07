#!/usr/bin/env python
# Multiple Sequence Alignment pairwise export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2020 Rodrigo Siqueira
from libcpp.string cimport string
from libcpp.vector cimport vector
from cartesian cimport c_cartesian2
from database cimport c_database

cdef extern from "pairwise/pairwise.cuh" namespace "msa::pairwise":
    # The score of the alignment of a sequence pair.
    # @since 0.1.1
    ctypedef float c_score "msa::pairwise::score"

    cdef struct c_configuration "msa::pairwise::configuration":
        pass

    # Manages all data and execution of the pairwise module.
    # @since 0.1.1
    cdef cppclass c_pairwise "msa::pairwise::manager":
        ctypedef c_score element_type

        c_pairwise()
        c_pairwise(c_pairwise&)

        c_pairwise& operator=(c_pairwise&) except +RuntimeError

        element_type& at "operator[]" (const c_cartesian2&) except +RuntimeError

        size_t count()

        @staticmethod
        c_pairwise run(c_configuration&) except +RuntimeError

    # The aminoacid substitution tables. These tables are stored contiguously
    # in memory, in order to facilitate accessing its elements.
    # @since 0.1.1
    cdef cppclass c_scoring_table "msa::pairwise::scoring_table":
        ctypedef c_score element_type

        element_type at "operator[]" (const c_cartesian2&)

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

    # Creates a module's configuration instance.
    cdef c_configuration configure(c_database&, string&, string&)

# Manages all data and execution of the pairwise module.
# @since 0.1.1
cdef class Pairwise:
    cdef c_pairwise thisptr

    @staticmethod
    cdef Pairwise wrap(c_pairwise&)

# Exposes a scoring table to Python's world.
# @since 0.1.1
cdef class ScoringTable:
    cdef c_scoring_table thisptr

    @staticmethod
    cdef ScoringTable wrap(c_scoring_table&)
