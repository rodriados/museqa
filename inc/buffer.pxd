#!/usr/bin/env python
# Multiple Sequence Alignment buffer cython export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2020 Rodrigo Siqueira
from libc.stdint cimport *
from libcpp.vector cimport vector

cdef extern from "buffer.hpp" namespace "msa" nogil:
    
    # Creates a general-purpose buffer. The buffer's idea is to store all of its
    # data contiguously in memory. Originally, the buffer is not growable.
    # @tparam T The buffer contents type.
    # @since 0.1.1
    cdef cppclass c_buffer "msa::buffer" [T]:
        ctypedef T element_type

        c_buffer()
        c_buffer(c_buffer&)

        c_buffer& operator=(c_buffer&) except +RuntimeError
        
        element_type& operator[](ptrdiff_t) except +RuntimeError

        element_type *begin()
        element_type *end()

        size_t size()

    # Manages a slice of a buffer. The buffer must have already been initialized
    # and will have boundaries checked according to slice pointers.
    # @tparam T The buffer contents type.
    # @since 0.1.1
    cdef cppclass c_slice_buffer "msa::slice_buffer" [T] (c_buffer[T]):
        c_slice_buffer()
        c_slice_buffer(c_slice_buffer&)
        c_slice_buffer(c_buffer[T]&, ptrdiff_t, size_t) except +RuntimeError
        c_slice_buffer(c_buffer[T]&, c_slice_buffer&) except +RuntimeError

        c_slice_buffer& operator=(c_slice_buffer&) except +RuntimeError

        ptrdiff_t displ()
