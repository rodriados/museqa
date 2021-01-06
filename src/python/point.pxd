#!/usr/bin/env python
# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file Wraps the software's point module and exposes it to Python world.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2020-present Rodrigo Siqueira
from libc.stdint cimport *

cdef extern from "point.hpp" namespace "museqa":
    # Represents a bi-dimensional point value.
    # @since 0.1.1
    cdef cppclass c_point2 "museqa::point2" [T]:
        ctypedef T dimension_type

        c_point2()
        c_point2(c_point2&)
        c_point2(T, T)

        c_point2& operator=(c_point2&)
        
        dimension_type at "operator[]" (ptrdiff_t) except +RuntimeError
