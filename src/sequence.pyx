#!/usr/bin/env python
# Multiple Sequence Alignment sequence wrapper file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2020 Rodrigo Siqueira
from libcpp.string cimport string
from sequence cimport c_sequence
from encoder cimport c_unit, c_decode

from functools import singledispatch

__all__ = ["Sequence"]

# Sequence wrapper. This class is responsible for interfacing all interactions between
# Python code to the underlying C++ sequence object.
# @since 0.1.1
cdef class Sequence:
    # Instantiates a new sequence.
    # @param contents The sequence contents.
    def __cinit__(self, contents):
        @singledispatch
        def overload(value):
            raise TypeError("unexpected argument type")

        @overload.register(bytes)
        def from_bytes(bytes value):
            return value

        overload.register(str, lambda value: from_bytes(value.encode('ascii')))
        
        cdef string value = overload(contents)
        self.c_set(c_sequence(value))

    # Gives access to a specific location in buffer's data.
    # @param offset The requested buffer offset.
    # @return The buffer's position pointer.
    def __getitem__(self, int offset):
        cdef c_unit result = self.thisptr.at(offset)
        return chr(c_decode(result))

    # Transforms the sequence into a string.
    # @return The sequence representation as a string.
    def __str__(self):
        cdef bytes contents = self.thisptr.decode()
        return contents.decode('ascii')

    # Informs the length of the sequence.
    # @return int The sequence's length.
    @property
    def length(self):
        return self.thisptr.length()
