#!/usr/bin/env python
# cython: language_level = 2
# Multiple Sequence Alignment parser export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2019 Rodrigo Siqueira
from libcpp.vector cimport vector
from libcpp.string cimport string
from database cimport cDatabaseEntry

cdef extern from "parser.hpp" namespace "parser":
    cdef vector[cDatabaseEntry] cparse "parser::parse"(const string&, const string&)
    cdef vector[cDatabaseEntry] cparse "parser::parse"(const string&)

    cdef vector[cDatabaseEntry] cparseMany "parser::parseMany"(const vector[string]&, const string&)
    cdef vector[cDatabaseEntry] cparseMany "parser::parseMany"(const vector[string]&)

cdef extern from "parser/fasta.hpp" namespace "parser":
    cdef vector[cDatabaseEntry] cfasta "parser::fasta"(const string&)
