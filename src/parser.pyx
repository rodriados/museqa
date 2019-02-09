#!/usr/bin/env python
# cython: language_level = 2
# Multiple Sequence Alignment parser wrapper file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2019 Rodrigo Siqueira
from libcpp.vector cimport vector
from libcpp.string cimport string
from database cimport cDatabaseEntry
from database import DatabaseEntry
from parser cimport *

# Converts a C++ database entry vector to a Python list.
# @param entries The entries vector to be converted.
# @return The vector converted into a list.
cdef object toList(vector[cDatabaseEntry]& entries):
    return [DatabaseEntry(d.description, d.sequence.toString()) for d in entries]

# Parses a list of files and produces a list of database entries.
# @param list filenames The list of files to parse.
# @param str ext Parse all files using this parser.
# @return list List containing all entries parsed from file.
def any(*filenames, **kwargs):
    cdef vector[string] files = filenames
    cdef string ext = str(kwargs.get("ext", str()))
    cdef vector[cDatabaseEntry] entries = cparseMany(files, ext)
    return toList(entries)

# Parses a FASTA file.
# @param str filename The file to be parsed.
# @return list List containing all entries parsed from file.
def fasta(str filename):
    cdef vector[cDatabaseEntry] entries = cfasta(filename)
    return toList(entries)
