#!/usr/bin/env python
# Multiple Sequence Alignment IO cython header file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2020 Rodrigo Siqueira
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "io/loader.hpp" namespace "msa::io" nogil:
    # Defines the generic object loader type. Whenever a new type may loaded
    # directly from a file, this struct must be specialized for given type.
    # @tparam T The target object type to load.
    # @since 0.1.1
    cdef cppclass c_loader "msa::io::loader" [T]:
        c_loader()
        c_loader(c_loader&)

        c_loader& operator=(c_loader&)

        T load(string&)
        T load(string&, string&)

        @staticmethod
        vector[string]& list()

cdef extern from "io/dumper.hpp" namespace "msa::io" nogil:
    # Defines the generic object dumper type. This struct must be specialized
    # to a new type, whenever a new dumpable type is introduced.
    # @tparam T The target object type to dump.
    # @since 0.1.1
    cdef cppclass c_dumper "msa::io::dumper" [T]:
        c_dumper()
        c_dumper(c_dumper&)

        c_dumper& operator=(c_dumper&)

        bool dump(T&, string&)
        bool dump(T&, string&, string&)

        @staticmethod
        vector[string]& list()
