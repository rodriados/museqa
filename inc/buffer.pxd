#!/usr/bin/env python
# Multiple Sequence Alignment buffer export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2020 Rodrigo Siqueira
from libc.stdint cimport *
from libcpp.vector cimport vector

cdef extern from "buffer.hpp" namespace "msa" nogil:
    # Creates a general-purpose buffer. The buffer's idea is to store all of its
    # data contiguously in memory. Originally, the buffer is not growable.
    # @tparam T The buffer contents type.
    # @since 0.1.1
    cdef cppclass cBuffer "msa::buffer" [T]:
        ctypedef T element_type

        cBuffer()
        cBuffer(cBuffer&)
        cBuffer(element_type *, size_t)

        cBuffer& operator=(cBuffer&) except +RuntimeError
        element_type& operator[](ptrdiff_t) except +RuntimeError

        element_type *begin()
        element_type *end()
        element_type *raw()

        size_t size()

        @staticmethod
        cBuffer copy(cBuffer&)
        @staticmethod
        cBuffer copy(vector[element_type]&)
        @staticmethod
        cBuffer copy(element_type *, size_t)
        @staticmethod
        cBuffer make()
        @staticmethod
        cBuffer make(size_t)

    # Manages a slice of a buffer. The buffer must have already been initialized
    # and will have boundaries checked according to slice pointers.
    # @tparam T The buffer contents type.
    # @since 0.1.1
    cdef cppclass cSliceBuffer "msa::slice_buffer" [T] (cBuffer[T]):
        cSliceBuffer()
        cSliceBuffer(cSliceBuffer&)
        cSliceBuffer(cBuffer[T]&, ptrdiff_t, size_t) except +RuntimeError
        cSliceBuffer(cBuffer[T]&, cSliceBuffer&) except +RuntimeError

        cSliceBuffer& operator=(cSliceBuffer&) except +RuntimeError

        ptrdiff_t displ()
