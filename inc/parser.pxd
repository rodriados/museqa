#!/usr/bin/env python
# Multiple Sequence Alignment parser export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2020 Rodrigo Siqueira
from libcpp.vector cimport vector
from libcpp.string cimport string
from database cimport *

cdef extern from "parser.hpp" namespace "msa::parser":
    cdef c_database c_parse "msa::parser::parse"(string&) except +RuntimeError
    cdef c_database c_parse "msa::parser::parse"(string&, string&) except +RuntimeError

    cdef c_database c_parse "msa::parser::parse"(vector[string]&) except +RuntimeError
    cdef c_database c_parse "msa::parser::parse"(vector[string]&, string&) except +RuntimeError

cdef extern from "parser/fasta.hpp" namespace "msa::parser":
    cdef c_database c_fasta "msa::parser::fasta"(string&) except +RuntimeError
