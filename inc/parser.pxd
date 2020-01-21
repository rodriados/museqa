#!/usr/bin/env python
# cython: language_level = 3
# Multiple Sequence Alignment parser export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2019 Rodrigo Siqueira
from libcpp.vector cimport vector
from libcpp.string cimport string
from database cimport cDatabaseEntry

cdef extern from "parser.hpp" namespace "parser":
    cdef vector[cDatabaseEntry] parse(const string&, const string&) except +RuntimeError
    cdef vector[cDatabaseEntry] parse(const string&) except +RuntimeError

    cdef vector[cDatabaseEntry] parse_many(const vector[string]&, const string&) except +RuntimeError
    cdef vector[cDatabaseEntry] parse_many(const vector[string]&) except +RuntimeError 

cdef extern from "parser/fasta.hpp" namespace "parser":
    cdef vector[cDatabaseEntry] parse_fasta "parser::fasta"(const string&) except +RuntimeError
