#!/usr/bin/env python
# Multiple Sequence Alignment sequence export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2020 Rodrigo Siqueira
from libc.stdint cimport *
from libcpp.string cimport string
from buffer cimport cSliceBuffer
from encoder cimport cUnit, cBlock
from encoder cimport cBuffer as cEncoderBuffer

cdef extern from "sequence.hpp" namespace "msa" nogil:
    # Holds an enconded sequence. The encoding pattern will used throughout all
    # steps: it saves up to a third of the required space and is easily revertable.
    # @since 0.1.1
    cdef cppclass cSequence "msa::sequence" (cEncoderBuffer):
        cSequence()
        cSequence(cSequence&)
        cSequence(cEncoderBuffer&)
        cSequence(char *, size_t)
        cSequence(string&)

        cSequence& operator=(cSequence&) except +RuntimeError
        cUnit at "operator[]" (ptrdiff_t) except +IndexError
        cBlock block(int) except +RuntimeError

        size_t length()
        string decode() except +RuntimeError

    # Manages a slice of a sequence. The sequence must have already been initialized
    # and will have boundaries checked according to view pointers.
    # @since 0.1.1
    cdef cppclass cSequenceView "msa::sequence_view" (cSliceBuffer[cBlock]):
        cSequenceView()
        cSequenceView(cSequenceView&)

        cSequenceView& operator=(cSequenceView&) except +RuntimeError
        cUnit at "operator[]" (ptrdiff_t) except +IndexError
        cBlock block(int) except +RuntimeError

        size_t length()
        string decode() except +RuntimeError

# Sequence wrapper. This class is responsible for interfacing all interactions between
# Python code to the underlying C++ sequence object.
# @since 0.1.1
cdef class Sequence:
    cdef cSequence thisptr
    
    @staticmethod
    cdef Sequence wrap(cSequence&)
