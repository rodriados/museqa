#!/usr/bin/env python
# cython: language_level = 3
# Multiple Sequence Alignment pairwise export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2019 Rodrigo Siqueira
from libc.stdint cimport *
from libcpp.string cimport string
from libcpp.vector cimport vector
from database cimport cDatabase

cdef extern from "pairwise/pairwise.cuh":

    # The score of the alignment of a sequence pair.
    # @since 0.1.1
    ctypedef int32_t cScore "pairwise::Score"

    # The aminoacid matches scoring tables are stored contiguously. Thus,
    # we explicitly declare their sizes.
    # @since 0.1.1
    ctypedef int8_t cScoringTable "pairwise::ScoringTable"[25][25]

    # Manages and encapsulates all configurable aspects of the pairwise module.
    # @since 0.1.1
    cdef struct cConfiguration "pairwise::Configuration":
        const cDatabase& db
        string algorithm
        string table

    # Manages all data and execution of the pairwise module.
    # @since 0.1.1
    cdef cppclass cPairwise "pairwise::Pairwise":
        cPairwise() except +
        cPairwise(const cPairwise&) except +

        cPairwise& operator=(const cPairwise&)
        cScore& operator[](ptrdiff_t) except +IndexError

        const cScore *getBuffer()
        size_t getSize()

        void run(const cConfiguration&) except +RuntimeError

    cdef cScoringTable *cgetTable "pairwise::table::get"(const string&) except +RuntimeError
    cdef const vector[string]& cgetTableList "pairwise::table::getList"()

    cdef cConfiguration configure "pairwise::configure"(const cDatabase&, const string&, const string&)

# Manages all data and execution of the pairwise module.
# @since 0.1.1
cdef class Pairwise:
    cdef cPairwise cRef

# Exposes a scoring table to Python world.
# @since 0.1.1
cdef class ScoringTable:
    cdef cScoringTable *cPtr;
