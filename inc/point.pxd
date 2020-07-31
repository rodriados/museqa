#!/usr/bin/env python
# Multiple Sequence Alignment point export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2020 Rodrigo Siqueira
from libc.stdint cimport *

cdef extern from "point.hpp" namespace "msa":
    # Represents a bi-dimensional point value.
    # @since 0.1.1
    cdef cppclass c_point2 "msa::point2" [T]:
        ctypedef T element_type

        c_point2()
        c_point2(c_point2&)
        c_point2(T, T)

        c_point2& operator=(c_point2&)
        
        element_type at "operator[]" (T) except +RuntimeError
