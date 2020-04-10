#!/usr/bin/env python
# Multiple Sequence Alignment sequence database export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2020 Rodrigo Siqueira
from libcpp.set cimport set
from libcpp.vector cimport vector
from libcpp.string cimport string
from sequence cimport cSequence

cdef extern from "database.hpp" namespace "msa" nogil:
    # Stores a list of sequences read from possibly different sources. The added
    # sequences can only be accessed via their respective identity or iterator.
    # @since 0.1.1
    cdef cppclass cDatabase "msa::database":
        ctypedef cSequence element_type

        cppclass entry_type:
            string description
            element_type contents

        cDatabase()
        cDatabase(cDatabase&) except +RuntimeError

        cDatabase& operator=(cDatabase&) except +RuntimeError
        entry_type& at "operator[]" (ptrdiff_t) except +IndexError
        entry_type& at "operator[]" (string&) except +IndexError

        void add(element_type&) except +RuntimeError
        void add(vector[element_type]&) except +RuntimeError
        void add(string&, element_type&) except +RuntimeError

        void merge(cDatabase&) except +RuntimeError

        cDatabase only(set[string]&) except +RuntimeError
        cDatabase only(set[ptrdiff_t]&) except +RuntimeError

        entry_type *begin()
        entry_type *end()

        size_t count()

# Database wrapper. This class is responsible for interfacing all interactions between
# Python code to the underlying C++ database object.
# @since 0.1.1
cdef class Database:
    cdef cDatabase thisptr

    @staticmethod
    cdef Database wrap(cDatabase&)
