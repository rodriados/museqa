#!/usr/bin/env python
# Multiple Sequence Alignment sequence database export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2020 Rodrigo Siqueira
from libcpp.set cimport set
from libcpp.vector cimport vector
from libcpp.string cimport string
from sequence cimport c_sequence

cdef extern from "database.hpp" namespace "msa" nogil:
    # Stores a list of sequences read from possibly different sources. The added
    # sequences can only be accessed via their respective identity or iterator.
    # @since 0.1.1
    cdef cppclass c_database "msa::database":
        ctypedef c_sequence element_type

        cppclass entry_type:
            string description
            element_type contents

        c_database()
        c_database(c_database&) except +RuntimeError

        c_database& operator=(c_database&) except +RuntimeError
        
        entry_type& at "operator[]" (ptrdiff_t) except +IndexError
        entry_type& at "operator[]" (string&) except +IndexError

        void add(element_type&) except +RuntimeError
        void add(vector[element_type]&) except +RuntimeError
        void add(string&, element_type&) except +RuntimeError

        void merge(c_database&) except +RuntimeError

        c_database only(set[string]&) except +RuntimeError
        c_database only(set[ptrdiff_t]&) except +RuntimeError

        entry_type *begin()
        entry_type *end()

        size_t count()

cdef extern from "io/loader/database.hpp" namespace "msa::io" nogil:
    # Imports the IO loader's specialization for databases. This will allow us to
    # use the exactly same loader we do in C++.
    pass

# Database wrapper. This class is responsible for interfacing all interactions between
# Python code to the underlying C++ database object.
# @since 0.1.1
cdef class Database:
    cdef c_database thisptr

    # Sets the underlying C++ object to the given target instance.
    # @param target The target object to use as underlying instance.
    cdef inline void c_set(self, const c_database& target):
        self.thisptr = target

    # Retrieves the instance's underlying C++ object.
    # @return The underlying C++ object instance.
    cdef inline c_database c_get(self):
        return self.thisptr

    # Wraps an existing database instance.
    # @param target The database to be wrapped.
    # @return The new wrapper instance.
    @staticmethod
    cdef inline Database wrap(const c_database& target):
        cdef Database wrapper = Database.__new__(Database)
        wrapper.c_set(target)
        return wrapper
