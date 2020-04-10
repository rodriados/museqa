#!/usr/bin/env python
# Multiple Sequence Alignment parser export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2020 Rodrigo Siqueira
from libcpp.vector cimport vector
from libcpp.string cimport string
from database cimport *

cdef extern from "parser.hpp" namespace "msa::parser":
    cdef cDatabase cparse "msa::parser::parse"(string&) except +RuntimeError
    cdef cDatabase cparse "msa::parser::parse"(string&, string&) except +RuntimeError

    cdef cDatabase cparse "msa::parser::parse"(vector[string]&) except +RuntimeError
    cdef cDatabase cparse "msa::parser::parse"(vector[string]&, string&) except +RuntimeError

cdef extern from "parser/fasta.hpp" namespace "msa::parser":
    cdef cDatabase cfasta "msa::parser::fasta"(string&) except +RuntimeError
