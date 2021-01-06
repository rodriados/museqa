#!/usr/bin/env python
# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file Implementation for the database module wrapper.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-present Rodrigo Siqueira
from libcpp.set cimport set
from libcpp.string cimport string
from database cimport c_database
from sequence cimport c_sequence, Sequence
from sequence import Sequence
from io cimport c_loader

from collections import namedtuple
from functools import singledispatch

__all__ = ["Database"]

# Database wrapper. This class is responsible for interfacing all interactions between
# Python code to the underlying C++ database object.
# @since 0.1.1
cdef class Database:
    # Represents an entry instance, composed of a description and the contents of
    # a sequence. This representation can be used create, store and access the database.
    # @since 0.1.1
    entry = namedtuple('entry', ['description', 'contents'])

    # Gives access to a specific sequence of the database.
    # @param key The requested sequence key or offset.
    # @return The requested sequence.
    def __getitem__(self, key):
        @singledispatch
        def overload(value):
            raise TypeError("could not find sequence")

        @overload.register(int)
        def from_offset(int value):
            cdef c_database.entry_type entry = self.thisptr.at(value)
            return Database.entry(entry.description.decode(), Sequence.wrap(entry.contents))

        @overload.register(bytes)
        def from_key(bytes value):
            cdef c_database.entry_type entry = self.thisptr.at(value)
            return Database.entry(entry.description.decode(), Sequence.wrap(entry.contents))

        overload.register(str, lambda value: from_key(value.encode()))
        return overload(key)

    # Adds new sequences to database.
    # @param target The new sequence to add to database.
    def add(self, target):
        @singledispatch
        def overload(value):
            raise TypeError("could not add sequence to database")

        @overload.register(bytes)
        def from_bytes(bytes value):
            self.thisptr.add(c_sequence(value))

        @overload.register(tuple)
        def from_tuple(tuple value):
            description, contents = value
            self.thisptr.add(description.encode(), Sequence(contents).c_get())

        @overload.register(dict)
        def from_dict(dict value):
            for entry in value.items():
                from_tuple(entry)

        @overload.register(Sequence)
        def from_sequence(Sequence value):
            self.thisptr.add(value.c_get())

        overload.register(list, lambda value: [overload(i) for i in value])
        overload.register(str,  lambda value: from_bytes(value.encode('ascii')))
        overload(target)

    # Adds all elements from another database into this instance.
    # @param other The database to merge into this instance.
    def merge(self, Database other):
        self.thisptr.merge(other.c_get())

    # Creates a new database with only a set of selected elements.
    # @param keys The entry's keys' to be included in new database.
    # @return The new database containing only the selected elements.
    def only(self, list keys):
        if not all([type(key) is type(keys[0]) for key in keys]):
            raise TypeError("cannot select sequences via different key types")

        @singledispatch
        def overload(*values):
            raise TypeError("could not select sequences")

        cdef c_database result

        @overload.register(bytes)
        def from_keys(*values):
            cdef set[string] selected = values
            result = self.thisptr.only(selected)

        @overload.register(int)
        def from_offsets(*values):
            cdef set[ptrdiff_t] selected = values
            result = self.thisptr.only(selected)

        overload.register(str, lambda *values: from_keys(*[val.encode() for val in values]))
        overload(*keys)

        return Database.wrap(result)

    # Loads a new database from the given files.
    # @param filenames The files to load a new database from.
    # @return The new database instance.
    @staticmethod
    def load(*filenames):
        cdef c_database result
        cdef c_loader[c_database] loader

        for filename in filenames:
            result.merge(loader.load(filename.encode()))

        return Database.wrap(result)

    # Informs the number of sequences in database.
    # @return int The total number of sequences in database.
    @property
    def count(self):
        return self.thisptr.count()
