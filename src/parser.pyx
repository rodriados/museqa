#!/usr/bin/env python
# cython: language_level = 2
# Multiple Sequence Alignment parser wrapper file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2019 Rodrigo Siqueira
from libcpp.vector cimport vector
from libcpp.string cimport string
from parser cimport cparseMany
from database cimport cDatabaseEntry
from database import Database, DatabaseEntry

# Parses a list of files and produces a list of database entries.
# @param list filenames The list of files to parse.
# @param dict kwargs The named arguments.
def parse(*filenames, **kwargs):
    cdef vector[string] files = filenames
    cdef string ext = str(kwargs.get("ext", str()))
    cdef vector[cDatabaseEntry] entries = cparseMany(files, ext)

    dlist = [DatabaseEntry(d.description, d.sequence.toString()) for d in entries]
    return Database(*dlist)
