#!/usr/bin/env python
# Multiple Sequence Alignment cartesian export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2020 Rodrigo Siqueira
from libc.stdint cimport *

cdef extern from "cartesian.hpp" namespace "msa":
    # Represents a bi-dimensional cartesian value, that can be used either as a
    # space a point or a vector representation.
    # @since 0.1.1
    cdef cppclass c_cartesian2 "msa::cartesian2" [T]:
        ctypedef T element_type

        c_cartesian2()
        c_cartesian2(c_cartesian2&)
        c_cartesian2(T, T)

        c_cartesian2& operator=(c_cartesian2&)
        
        element_type at "operator[]" (T) except +RuntimeError

        c_cartesian2 operator+(c_cartesian2&)
        c_cartesian2 operator*(int)

        element_type collapse(c_cartesian2&)
        element_type volume()
