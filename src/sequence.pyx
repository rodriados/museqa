#!/usr/bin/env python
# Multiple Sequence Alignment sequence wrapper file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2020 Rodrigo Siqueira
from libcpp.string cimport string
from sequence cimport cSequence
from encoder cimport cdecode

from functools import singledispatch

# Sequence wrapper. This class is responsible for interfacing all interactions between
# Python code to the underlying C++ sequence object.
# @since 0.1.1
cdef class Sequence:
    # Instantiates a new sequence.
    # @param contents The sequence contents.
    def __cinit__(self, contents = bytes()):
        @singledispatch
        def default(value):
            raise TypeError("could not create sequence")

        @default.register(bytes)
        def fromBytes(bytes value):
            self.thisptr = cSequence(value)

        @default.register(Sequence)
        def fromSelf(Sequence value):
            self.thisptr = value.thisptr

        default.register(str, lambda value: fromBytes(value.encode('ascii')))
        default(contents)

    # Gives access to a specific location in buffer's data.
    # @param offset The requested buffer offset.
    # @return The buffer's position pointer.
    def __getitem__(self, int offset):
        return chr(cdecode(self.thisptr.at(offset)))

    # Transforms the sequence into a string.
    # @return The sequence representation as a string.
    def __str__(self):
        contents = self.thisptr.decode()
        return contents.decode('ascii')

    # Wraps an existing sequence instance.
    # @param target The sequence to be wrapped.
    # @return The new wrapper instance.
    @staticmethod
    cdef Sequence wrap(cSequence& target):
        instance = <Sequence>Sequence.__new__(Sequence)
        instance.thisptr = target
        return instance

    # Informs the length of the sequence.
    # @return int The sequence's length.
    @property
    def length(self):
        return self.thisptr.length()
