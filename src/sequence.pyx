#!/usr/bin/env python
# cython: language_level = 2
# Multiple Sequence Alignment sequence wrapper file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018 Rodrigo Siqueira
from sequence cimport cSequence

# Creates an sequence. This sequence is a buffer an any modification to
# it shall be implemented by inherited methods.
# @since 0.1.alpha
cdef class Sequence:

    # Instantiates a new sequence.
    # @param mixed source The string data source.
    def __cinit__(self, source):
        if isinstance(source, basestring):
            self._refSequence = cSequence(<string>source)
        else:
            self._refSequence = <const cSequence&>source._refSequence

    # Gives access to a specific location in buffer's data.
    # @param offset The requested buffer offset.
    # @return char The buffer's position pointer.
    def __getitem__(self, int offset):
        return str(unichr(self._refSequence[offset]))

    @property
    # Informs the length of the sequence.
    # @return int The sequence's length.
    def length(self):
        return self._refSequence.getLength()
