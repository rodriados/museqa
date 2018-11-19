#!/usr/bin/env python
# cython: language_level = 2
# Multiple Sequence Alignment fasta wrapper file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018 Rodrigo Siqueira
from libcpp.string cimport string
from fasta cimport cFasta, cFastaSequence
from sequence cimport Sequence

# Represents a sequence read from a fasta file.
# @since 0.1.alpha
cdef class FastaSequence(Sequence):

    # Instantiates a new fasta sequence.
    # @param str contents The sequence contents.
    # @param str description The sequence description.
    def __cinit__(self, contents, description = None):
        self._description = <string>(str(description))

    @property
    # Retrieves the description linked to the sequence.
    # @return The sequence's description.
    def description(self):
        return self._description

# Creates a list of sequences read from a fasta file. This sequence list
# is responsible for keeping track of sequences within its scope. Once a
# sequence is put into the list, it cannot leave.
# @since 0.1.alpha
cdef class Fasta:

    # Instantiates a new fasta file sequence list.
    # @param str filename The Fasta sequences file name.
    def __cinit__(self, filename = None):
        if filename is not None:
            filename = str(filename)
            self._wrapped.load(filename)
            self._filename = <string>filename

    # Gives access to a specific sequence of the list.
    # @param int offset The requested sequence offset.
    # @return FastaSequence The requested sequence.
    def __getitem__(self, int offset):
        cdef cFastaSequence sequence = self._wrapped[offset]
        return FastaSequence(sequence.toString(), sequence.getDescription())

    # Reads a file and allocates memory to all sequences contained in it.
    # @param str fname The name of the file to be loaded.
    def load(self, filename):
        filename = str(filename)
        self._wrapped.load(filename)

    # Pushes a new sequence into the list.
    # @param str contents The sequence contents.
    # @param str description The new sequence description.
    def push(self, contents, description = None):
        if isinstance(contents, FastaSequence):
            self._wrapped.push(contents.description, str(contents))
        else:
            self._wrapped.push(str(description), str(contents))

    @property
    # Informs the number of sequences in the list.
    # @return int The list's number of sequences.
    def count(self):
        return self._wrapped.getCount()
