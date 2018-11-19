#!/usr/bin/env python
# cython: language_level = 2
# Multiple Sequence Alignment fasta export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018 Rodrigo Siqueira
from libcpp.string cimport string
from sequence cimport cSequence, Sequence

cdef extern from "fasta.hpp":

    # Represents a sequence read from a fasta file.
    # @since 0.1.alpha
    cdef cppclass cFastaSequence "FastaSequence"(cSequence):
        cFastaSequence() except +
        cFastaSequence(const cFastaSequence&) except +
        cFastaSequence(const string&, const string&) except +
        cFastaSequence(const string&, const cSequence&) except +
        cFastaSequence(const string&, const char *, long) except +

        cFastaSequence& operator=(const cFastaSequence&)

        string& getDescription()

    # Creates a list of sequences read from a fasta file. This sequence list
    # is responsible for keeping track of sequences within its scope. Once a
    # sequence is put into the list, it cannot leave.
    # @since 0.1.alpha
    cdef cppclass cFasta "Fasta":
        cFasta() except +
        cFasta(const cFasta&) except +
        cFasta(const string&) except +IOError

        cFasta& operator=(const cFasta&)
        const cFastaSequence& operator[](long)

        long getCount()

        void load(const string&) except +IOError
        void push(const cFastaSequence&)
        void push(const string&, const string&)
        void push(const string&, const char *, long)

# Represents a sequence read from a fasta file.
# @since 0.1.alpha
cdef class FastaSequence(Sequence):

    # The fasta sequence description.
    cdef string _description

# Creates a list of sequences read from a fasta file. This sequence list
# is responsible for keeping track of sequences within its scope. Once a
# sequence is put into the list, it cannot leave.
# @since 0.1.alpha
cdef class Fasta:

    # Reference for the instance of the wrapped C++ object.
    cdef cFasta _wrapped

    # The current fasta file name.
    cdef string _filename
