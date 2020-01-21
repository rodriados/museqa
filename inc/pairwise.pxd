#!/usr/bin/env python
# cython: language_level = 3
# Multiple Sequence Alignment pairwise export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2019 Rodrigo Siqueira
from libc.stdint cimport *
from libcpp.string cimport string
from libcpp.vector cimport vector
from database cimport cDatabase

cdef extern from "pairwise/pairwise.cuh" namespace "pairwise":

    # The score of the alignment of a sequence pair.
    # @since 0.1.1
    ctypedef float cScore "pairwise::score"

    # Manages and encapsulates all configurable aspects of the pairwise module.
    # @since 0.1.1
    cdef struct cConfiguration "pairwise::configuration":
        const cDatabase* db
        string algorithm
        string table

    # Manages all data and execution of the pairwise module.
    # @since 0.1.1
    cdef cppclass cPairwise "pairwise::manager":
        cPairwise() except +
        cPairwise(const cPairwise&) except +

        cPairwise& operator=(const cPairwise&)
        cScore operator[](const cCartesian[2]&) except +RuntimeError

        size_t count()

        @staticmethod
        cPairwise run(const cConfiguration&) except +RuntimeError

    # The aminoacid substitution tables. These tables are stored contiguously
    # in memory, in order to facilitate accessing its elements.
    # @since 0.1.1
    cdef cppclass cScoringTable "pairwise::scoring_table":
        cScore operator[](const cCartesian[2]&)

        cScore penalty()

        @staticmethod
        cScoringTable make(const string&) except +RuntimeError

        @staticmethod
        const vector[string]& list()

    # Creates a module's configuration instance.
    cdef cConfiguration configure(const cDatabase&, const string&, const string&)

# Manages all data and execution of the pairwise module.
# @since 0.1.1
cdef class Pairwise:
    cdef cPairwise cRef

# Exposes a scoring table to Python world.
# @since 0.1.1
cdef class ScoringTable:
    cdef cScoringTable cRef;
