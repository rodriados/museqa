#!/usr/bin/env python
# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file Wraps the software's encoder module and exposes it to Python world.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2020-present Rodrigo Siqueira
from libcpp.string cimport string
from libc.stdint cimport *

cdef extern from "encoder.hpp" namespace "museqa::encoder" nogil:
    # The encoder character or unit type.
    # @since 0.1.1
    ctypedef uint8_t c_unit "museqa::encoder::unit"

    # The encoder sequence block type.
    # @since 0.1.1
    ctypedef uint16_t c_block "museqa::encoder::block"

    cdef c_unit c_encode "museqa::encoder::encode" (char)
    cdef char c_decode "museqa::encoder::decode" (c_unit) except +RuntimeError
