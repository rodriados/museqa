#!/usr/bin/env python
# Multiple Sequence Alignment pairwise export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2020 Rodrigo Siqueira
from libcpp.string cimport string
from libcpp.vector cimport vector
from cartesian cimport cCartesian
from database cimport cDatabase
from buffer cimport cBuffer

cdef extern from "pairwise/pairwise.cuh" namespace "msa::pairwise":

    # The score of the alignment of a sequence pair.
    # @since 0.1.1
    ctypedef float cScore "msa::pairwise::score"

    # Manages and encapsulates all configurable aspects of the pairwise module.
    # @since 0.1.1
    cdef struct cConfiguration "msa::pairwise::configuration":
        const cDatabase *db
        string algorithm
        string table

    # Manages all data and execution of the pairwise module.
    # @since 0.1.1
    cdef cppclass cPairwise "msa::pairwise::manager" (cBuffer[cScore]):
        ctypedef cScore element_type

        cPairwise()
        cPairwise(cPairwise&)

        cPairwise& operator=(cPairwise&) except +RuntimeError
        element_type at "operator[]" (cCartesian&) except +RuntimeError

        size_t count()

        @staticmethod
        cPairwise run(cConfiguration&) except +RuntimeError

    # The aminoacid substitution tables. These tables are stored contiguously
    # in memory, in order to facilitate accessing its elements.
    # @since 0.1.1
    cdef cppclass cScoringTable "msa::pairwise::scoring_table":
        ctypedef cScore element_type

        element_type at "operator[]" (cCartesian&)
        element_type penalty()

        @staticmethod
        cScoringTable make(string&) except +RuntimeError

        @staticmethod
        vector[string]& list()

    # Creates a module's configuration instance.
    cdef cConfiguration configure(cDatabase&, string&, string&)

# Manages all data and execution of the pairwise module.
# @since 0.1.1
cdef class Pairwise:
    cdef cPairwise thisptr

    @staticmethod
    cdef Pairwise wrap(cPairwise&)

# Exposes a scoring table to Python's world.
# @since 0.1.1
cdef class ScoringTable:
    cdef cScoringTable thisptr

    @staticmethod
    cdef ScoringTable wrap(cScoringTable&)
