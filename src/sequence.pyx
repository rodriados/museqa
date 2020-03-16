#!/usr/bin/env python
# Multiple Sequence Alignment sequence wrapper file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2020 Rodrigo Siqueira
from libcpp.string cimport string
from sequence cimport c_sequence
from encoder cimport c_decode

# Sequence wrapper. This class is responsible for interfacing all interactions between
# Python code to the underlying C++ sequence object.
# @since 0.1.1
cdef class sequence:
    # Instantiates a new sequence.
    # @param contents The sequence contents.
    def __cinit__(self, str contents = str()):
        cdef string value = bytes(contents, encoding = 'ascii')
        self.thisptr = c_sequence(value)

    # Gives access to a specific location in buffer's data.
    # @param offset The requested buffer offset.
    # @return The buffer's position pointer.
    def __getitem__(self, int offset):
        return chr(c_decode(self.thisptr.at(offset)))

    # Transforms the sequence into a string.
    # @return The sequence representation as a string.
    def __str__(self):
        contents = self.thisptr.decode()
        return contents.decode('ascii')

    # Wraps an existing sequence instance.
    # @param target The sequence to be wrapped.
    @staticmethod
    cdef object wrap(c_sequence& target):
        instance = sequence()
        instance.thisptr = target
        return instance

    # Informs the length of the sequence.
    # @return int The sequence's length.
    @property
    def length(self):
        return self.thisptr.length()
