#!/usr/bin/env python
# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file Implementation for the sequence module wrapper.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-present Rodrigo Siqueira
from libcpp.string cimport string
from encoder cimport c_unit, c_encode, c_decode, c_end
from sequence cimport c_sequence

from functools import singledispatch

__all__ = [
    "Sequence"
,   "encode"
]

# Sequence wrapper. This class is responsible for interfacing all interactions between
# Python code to the underlying C++ sequence object.
# @since 0.1.1
cdef class Sequence:
    # The character to indicate padding after a sequence has already been finished.
    # @since 0.1.1
    padding = chr(c_decode(c_end))

    # Instantiates a new sequence.
    # @param contents The sequence contents.
    def __cinit__(self, contents = bytes()):
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

# Encodes a character string into the project's internal sequence format.
# @param sequence The sequence to be encoded.
# @return The encoded sequence.
def encode(sequence):
    units = [c_encode(ord(unit)) for unit in sequence]
    units = [chr(c_decode(unit)) for unit in units]

    padding = len(sequence) % 3
    padding = [Sequence.padding] * ((3 - padding) if padding > 0 else 0)

    return str.join("", units + padding)
