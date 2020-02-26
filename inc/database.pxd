#!/usr/bin/env python
# cython: language_level = 3
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
    cdef struct cDatabaseEntry "database_entry":
        string description
        cSequence raw_sequence

    # Stores a list of sequences read from possible different sources. The
    # sequences may be identified by description or inclusion index.
    # @since 0.1.1
    cdef cppclass cDatabase "database":
        cDatabase() except +
        cDatabase(const cDatabase&) except +

        cDatabase& operator=(const cDatabase&)
        const cSequence& operator[](ptrdiff_t) except +IndexError

        size_t count()
        const cDatabaseEntry& entry(ptrdiff_t) except +IndexError

        void add(const cSequence&)
        void add(const cDatabaseEntry&)
        void add(const string&, const cSequence&)
        void add_many(const cDatabase&)
        void add_many(const vector[cSequence]&)
        void add_many(const vector[cDatabaseEntry]&)

        cDatabase excluding "except"(const set[ptrdiff_t]&) except +IndexError
        cDatabase excluding "except"(const vector[ptrdiff_t]&) except +IndexError

        cDatabase only(const set[ptrdiff_t]&) except +IndexError
        cDatabase only(const vector[ptrdiff_t]&) except +IndexError

        void remove(ptrdiff_t) except +IndexError
        void remove_many(const set[ptrdiff_t]&) except +IndexError
        void remove_many(const vector[ptrdiff_t]&) except +IndexError

# Represents a sequence stored in database.
# @since 0.1.1
cdef class DatabaseEntry:
    cdef cDatabaseEntry cRef

# Stores a list of sequences read from possible different sources. The
# sequences may be identified by description or inclusion index.
# @since 0.1.1
cdef class Database:
    cdef cDatabase cRef

    cdef void __add_from_database(self, Database)
    cdef void __add_from_databaseEntry(self, DatabaseEntry)
    cdef void __add_from_sequence(self, Sequence)
    cdef void __add_from_string(self, string)
    cdef void __add_from_tuple(self, string, string)
