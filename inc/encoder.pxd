#!/usr/bin/env python
# Multiple Sequence Alignment encoder export file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2020 Rodrigo Siqueira
from libc.stdint cimport *
from libcpp.string cimport string
from buffer cimport cBuffer as cBaseBuffer

cdef extern from "encoder.hpp" namespace "msa::encoder" nogil:
    # The encoder character or unit type.
    # @since 0.1.1
    ctypedef uint8_t cUnit "msa::encoder::unit"

    # The encoder sequence block type.
    # @since 0.1.1
    ctypedef uint16_t cBlock "msa::encoder::block"

    # Aliases a block buffer into a more readable name.
    # @since 0.1.1
    ctypedef cBaseBuffer[cBlock] cBuffer "msa::encoder::buffer"

    cdef cUnit cencode "msa::encoder::encode" (char)
    cdef cBuffer cencode "msa::encoder::encode" (char *, size_t) except +RuntimeError

    cdef char cdecode "msa::encoder::decode" (cUnit) except +RuntimeError
    cdef string cdecode "msa::encoder::decode" (cBuffer&) except +RuntimeError
