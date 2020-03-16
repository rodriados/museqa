#!/usr/bin/env python
# Multiple Sequence Alignment sequence export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2020 Rodrigo Siqueira
from libc.stdint cimport *
from libcpp.string cimport string
from buffer cimport c_slice_buffer
from encoder cimport c_unit, c_block
from encoder cimport c_buffer as c_encoder_buffer

cdef extern from "sequence.hpp" namespace "msa" nogil:
    # Holds an enconded sequence. The encoding pattern will used throughout all
    # steps: it saves up to a third of the required space and is easily revertable.
    # @since 0.1.1
    cdef cppclass c_sequence "msa::sequence" (c_encoder_buffer):
        c_sequence()
        c_sequence(c_sequence&)
        c_sequence(c_encoder_buffer&)
        c_sequence(char *, size_t)
        c_sequence(string&)

        c_sequence& operator=(c_sequence&) except +RuntimeError
        c_unit at "operator[]" (ptrdiff_t) except +RuntimeError
        c_block block(int) except +RuntimeError

        size_t length()
        string decode() except +RuntimeError

    # Manages a slice of a sequence. The sequence must have already been initialized
    # and will have boundaries checked according to view pointers.
    # @since 0.1.1
    cdef cppclass c_sequence_view "msa::sequence_view" (c_slice_buffer[c_block]):
        c_sequence_view()
        c_sequence_view(c_sequence_view&)

        c_sequence_view& operator=(c_sequence_view&) except +RuntimeError
        c_unit operator[](ptrdiff_t) except +RuntimeError
        c_block block(int) except +RuntimeError

        size_t length()
        string decode() except +RuntimeError

# Sequence wrapper. This class is responsible for interfacing all interactions between
# Python code to the underlying C++ sequence object.
# @since 0.1.1
cdef class sequence:
    cdef c_sequence thisptr
    @staticmethod
    cdef object wrap(c_sequence&)
