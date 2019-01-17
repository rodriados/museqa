#!/usr/bin/env python
# cython: language_level = 2
# Multiple Sequence Alignment buffer export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2019 Rodrigo Siqueira
from libcpp.vector cimport vector

cdef extern from "buffer.hpp":

    # The base of a buffer class.
    # @since 0.1.alpha
    cdef cppclass cBaseBuffer "BaseBuffer"[T]:
        cBaseBuffer() except +
        cBaseBuffer(const cBaseBuffer[T]&) except +
        cBaseBuffer(T*, long) except +

        cBaseBuffer[T]& operator=(const cBaseBuffer[T]&)
        const T& operator[](long)

        T *getBuffer()
        long getSize()

    # Creates a general-purpose buffer. The buffer created is constant and
    # cannot be changed. For mutable buffers, please use strings or vectors.
    # @since 0.1.alpha
    cdef cppclass cBuffer "Buffer"[T](cBaseBuffer[T]):
        cBuffer() except +
        cBuffer(const cBuffer[T]&) except +
        cBuffer(const cBaseBuffer[T]&) except +
        cBuffer(const T *, long) except +
        cBuffer(const vector[T]&) except +

        cBuffer[T]& operator=(const cBuffer[T]&)

    # Represents a slice of a buffer.
    # @since 0.1.alpha
    cdef cppclass cBufferSlice "BufferSlice"[T](cBaseBuffer[T]):
        cBufferSlice() except +
        cBufferSlice(const cBufferSlice[T]&) except +
        cBufferSlice(const cBaseBuffer[T]&, long, long) except +
        cBufferSlice(const cBaseBuffer[T]&, const cBufferSlice[T]&) except +

        cBufferSlice[T]& operator=(const cBufferSlice[T]&)

        long getDispl()
