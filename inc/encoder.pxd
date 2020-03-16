#!/usr/bin/env python
# Multiple Sequence Alignment encoder export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2020 Rodrigo Siqueira
from libc.stdint cimport *
from libcpp.string cimport string
from buffer cimport c_buffer as c_base_buffer

cdef extern from "encoder.hpp" namespace "msa::encoder" nogil:
    # The encoder character or unit type.
    # @since 0.1.1
    ctypedef uint8_t c_unit "msa::encoder::unit"

    # The encoder sequence block type.
    # @since 0.1.1
    ctypedef uint16_t c_block "msa::encoder::block"

    # Aliases a block buffer into a more readable name.
    # @since 0.1.1
    ctypedef c_base_buffer[c_block] c_buffer "msa::encoder::buffer"

    cdef c_unit c_encode "msa::encoder::encode" (char)
    cdef c_buffer c_encode "msa::encoder::encode" (char *, size_t) except +RuntimeError

    cdef char c_decode "msa::encoder::decode" (c_unit) except +RuntimeError
    cdef string c_decode "msa::encoder::decode" (c_buffer&) except +RuntimeError

cdef extern from "encoder.cpp":
    pass
