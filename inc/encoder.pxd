#!/usr/bin/env python
# Multiple Sequence Alignment encoder export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2020 Rodrigo Siqueira
from libc.stdint cimport *
from libcpp.string cimport string

cdef extern from "encoder.hpp" namespace "msa::encoder" nogil:
    # The encoder character or unit type.
    # @since 0.1.1
    ctypedef uint8_t c_unit "msa::encoder::unit"

    # The encoder sequence block type.
    # @since 0.1.1
    ctypedef uint16_t c_block "msa::encoder::block"

    cdef c_unit c_encode "msa::encoder::encode" (char)
    cdef char c_decode "msa::encoder::decode" (c_unit) except +RuntimeError
