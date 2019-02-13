#!/usr/bin/env python
# cython: language_level = 2
# Multiple Sequence Alignment sequence database export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2019 Rodrigo Siqueira
from libcpp.set cimport set
from libcpp.vector cimport vector
from libcpp.string cimport string
from sequence cimport cSequence, Sequence

cdef extern from "database.hpp":

    # Allows sequences to be stored alongside their properties.
    # @since 0.1.1
    cdef struct cDatabaseEntry "DatabaseEntry":
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

        size_t getCount()
        const cDatabaseEntry& getEntry(ptrdiff_t) except +IndexError

        void add(const cSequence&)
        void add(const cDatabaseEntry&)
        void add(const string&, const cSequence&)
        void addMany(const cDatabase&)
        void addMany(const vector[cSequence]&)
        void addMany(const vector[cDatabaseEntry]&)

        cDatabase excluding "except"(const set[ptrdiff_t]&) except +IndexError
        cDatabase excluding "except"(const vector[ptrdiff_t]&) except +IndexError

        cDatabase only(const set[ptrdiff_t]&) except +IndexError
        cDatabase only(const vector[ptrdiff_t]&) except +IndexError

        void remove(ptrdiff_t) except +IndexError
        void removeMany(const set[ptrdiff_t]&) except +IndexError
        void removeMany(const vector[ptrdiff_t]&) except +IndexError

# Represents a sequence stored in database.
# @since 0.1.1
cdef class DatabaseEntry:
    cdef cDatabaseEntry cRef

# Stores a list of sequences read from possible different sources. The
# sequences may be identified by description or inclusion index.
# @since 0.1.1
cdef class Database:
    cdef cDatabase cRef

    cdef void __addFromDatabase(self, Database)
    cdef void __addFromDatabaseEntry(self, DatabaseEntry)
    cdef void __addFromSequence(self, Sequence)
    cdef void __addFromString(self, string)
    cdef void __addFromTuple(self, string, string)
