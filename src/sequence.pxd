#!/usr/bin/env python
# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file Wraps the software's sequence module and exposes it to Python world.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-present Rodrigo Siqueira
from encoder cimport c_unit, c_block
from libcpp.string cimport string
from libc.stdint cimport *

cdef extern from "sequence.hpp" namespace "museqa" nogil:
    # Holds an enconded sequence. The encoding pattern will used throughout all
    # steps: it saves up to a third of the required space and is easily revertable.
    # @since 0.1.1
    cdef cppclass c_sequence "museqa::sequence":
        c_sequence()
        c_sequence(c_sequence&)
        c_sequence(char *, size_t)
        c_sequence(string&)

        c_sequence& operator=(c_sequence&) except +RuntimeError

        c_unit at "operator[]" (ptrdiff_t) except +IndexError
        c_block block(int) except +RuntimeError

        string decode() except +RuntimeError
        size_t length()

    # Manages a slice of a sequence. The sequence must have already been initialized
    # and will have boundaries checked according to view pointers.
    # @since 0.1.1
    cdef cppclass c_sequence_view "museqa::sequence_view":
        c_sequence_view()
        c_sequence_view(c_sequence_view&)

        c_sequence_view& operator=(c_sequence_view&) except +RuntimeError

        c_unit at "operator[]" (ptrdiff_t) except +IndexError
        c_block block(int) except +RuntimeError

        string decode() except +RuntimeError
        size_t length()

# Sequence wrapper. This class interfaces all interactions between Python code to
# the underlying C++ sequence object.
# @since 0.1.1
cdef class Sequence:
    cdef c_sequence thisptr

    # Sets the underlying C++ object to the given target instance.
    # @param target The target object to use as underlying instance.
    cdef inline void c_set(self, const c_sequence& target):
        self.thisptr = target

    # Retrieves the instance's underlying C++ object.
    # @return The underlying C++ object instance.
    cdef inline c_sequence c_get(self):
        return self.thisptr

    # Wraps an existing sequence instance.
    # @param target The sequence to be wrapped.
    # @return The new wrapper instance.
    @staticmethod
    cdef inline Sequence wrap(const c_sequence& target):
        cdef Sequence wrapper = Sequence.__new__(Sequence)
        wrapper.c_set(target)
        return wrapper
