#!/usr/bin/env python
# cython: language_level = 2
# Multiple Sequence Alignment sequence export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2019 Rodrigo Siqueira
from libcpp.string cimport string

cdef extern from "sequence.hpp":

    # Creates an sequence. This sequence is a buffer an any modification to
    # it shall be implemented by inherited methods.
    # @since 0.1.alpha
    cdef cppclass cSequence "Sequence":
        cSequence() except +
        cSequence(const cSequence&) except +
        cSequence(const string&) except +

        cSequence& operator=(const cSequence&)
        const char& operator[](long)

        long getLength()
        string toString()

# Creates an sequence. This sequence is a buffer an any modification to
# it shall be implemented by inherited methods.
# @since 0.1.alpha
cdef class Sequence:

    # Reference for the instance of the wrapped C++ object.
    cdef cSequence _ref
