#!/usr/bin/env python
# cython: language_level = 3
# Multiple Sequence Alignment sequence wrapper file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2019 Rodrigo Siqueira
from libcpp.string cimport string
from sequence cimport cdecode, cSequence

# Creates an sequence. This sequence is a buffer an any modification to
# it shall be implemented by inherited methods.
# @since 0.1.1
cdef class Sequence:

    # Instantiates a new sequence.
    # @param str contents The string contents.
    def __cinit__(self, str contents = str()):
        cdef string value = bytes(contents, encoding = 'utf-8')
        self.cRef = cSequence(value)

    # Gives access to a specific location in buffer's data.
    # @param offset The requested buffer offset.
    # @return char The buffer's position pointer.
    def __getitem__(self, int offset):
        return chr(cdecode(self.cRef[offset]))

    # Transforms the sequence into a string.
    # @return The sequence representation as a string.
    def __str__(self):
        return self.cRef.toString().decode()

    # Informs the length of the sequence.
    # @return int The sequence's length.
    @property
    def length(self):
        return self.cRef.getLength()
