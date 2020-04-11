#!/usr/bin/env python
# Multiple Sequence Alignment cartesian export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2020 Rodrigo Siqueira
from libc.stdint cimport *

cdef extern from "cartesian.hpp" namespace "msa":
    # Represents abidimensional cartesian value, that can be used either as a space
    # a point or a vector representation.
    # @since 0.1.1
    cdef cppclass cCartesian "msa::cartesian<2>":
        ctypedef ptrdiff_t element_type

        cCartesian()
        cCartesian(cCartesian&)

        cCartesian(ptrdiff_t, ptrdiff_t)

        cCartesian& operator=(cCartesian&)
        element_type at "operator[]" (ptrdiff_t) except +RuntimeError
        cCartesian operator+(cCartesian&)
        cCartesian operator*(int)

        element_type collapse(cCartesian&)
        element_type volume()
