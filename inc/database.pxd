#!/usr/bin/env python
# cython: language_level = 2
# Multiple Sequence Alignment sequence database export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2019 Rodrigo Siqueira
from libcpp.string cimport string
from sequence cimport cSequence, Sequence

cdef extern from "database.hpp":

    # Allows sequences to be stored alongside their properties.
    # @since 0.1.1
    cdef cppclass cDatabaseEntry "DatabaseEntry":
        string description
        cSequence sequence

    # Stores a list of sequences read from possible different sources. The
    # sequences may be identified by description or inclusion index.
    # @since 0.1.1
    cdef cppclass cDatabase "Database":
        cDatabase() except +
        cDatabase(const cDatabase&) except +

        cDatabase& operator=(const cDatabase&)
        const cSequence& operator[](ptrdiff_t) except +IndexError

        void add(const cSequence&)
        void add(const cDatabaseEntry&)
        void add(const string&, const cSequence&)

        size_t getCount()
        const cDatabaseEntry& getEntry(ptrdiff_t) except +IndexError

# Stores a list of sequences read from possible different sources. The
# sequences may be identified by description or inclusion index.
# @since 0.1.1
cdef class Database:
    cdef cDatabase _ref

    cdef void _addFromSequence(self, Sequence)
    cdef void _addFromString(self, string)
    cdef void _addFromTuple(self, string, string)
