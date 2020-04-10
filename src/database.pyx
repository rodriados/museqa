#!/usr/bin/env python
# cython: language_level = 3
# Multiple Sequence Alignment sequence database wrapper file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2019 Rodrigo Siqueira
from libcpp.set cimport set
from libcpp.string cimport string
from database cimport cDatabase
from sequence cimport cSequence, Sequence

from collections import namedtuple
from functools import singledispatch

# Database wrapper. This class is responsible for interfacing all interactions between
# Python code to the underlying C++ database object.
# @since 0.1.1
cdef class Database:
    # Represents an entry instance, composed of a description and the contents of
    # a sequence. This representation can be used create, store and access the database.
    # @since 0.1.1
    entry = namedtuple('entry', ['description', 'contents'])

    # Gives access to a specific sequence of the database.
    # @param keyOrIndex The requested sequence key or offset.
    # @return The requested sequence.
    def __getitem__(self, keyOrIndex):
        @singledispatch
        def default(value):
            raise TypeError("could not get sequence")

        @default.register(int)
        def fromOffset(int value):
            cdef cDatabase.entry_type entry = self.thisptr.at(value)
            return Database.entry(entry.description.decode(), Sequence.wrap(entry.contents))

        @default.register(bytes)
        def fromKey(bytes value):
            cdef cDatabase.entry_type entry = self.thisptr.at(value)
            return Database.entry(entry.description.decode(), Sequence.wrap(entry.contents))

        default.register(str, lambda value: fromKey(value.encode()))
        return default(keyOrIndex)

    # Adds new sequences to database.
    # @param newSequence The new sequence(s) to be added.
    def add(self, newSequence):
        @singledispatch
        def default(value):
            raise TypeError("could not add sequence")

        @default.register(bytes)
        def fromBytes(bytes value):
            self.thisptr.add(cSequence(value))

        @default.register(tuple)
        def fromTuple(tuple value):
            description, contents = value
            self.thisptr.add(description.encode(), Sequence(contents).thisptr)

        @default.register(dict)
        def fromDict(dict value):
            [fromTuple(entry) for entry in value.items()]

        @default.register(Sequence)
        def fromSequence(Sequence value):
            self.thisptr.add(value.thisptr)

        default.register(list, lambda value: [default(i) for i in value])
        default.register(str,  lambda value: fromBytes(value.encode('ascii')))
        default(newSequence)

    # Adds all elements from another database into this instance.
    # @param other The database to merge into this instance.
    def merge(self, Database other):
        self.thisptr.merge(other.thisptr)

    # Creates a new database with only a set of selected elements.
    # @param entries The entries to be included in new database.
    # @return The new database containing only the selected elements.
    def only(self, list entries):
        if not all([type(entry) is type(entries[0]) for entry in entries]):
            raise TypeError("could select sequences via different key types")

        @singledispatch
        def default(*values):
            raise TypeError("could not select sequences")

        @default.register(bytes)
        def fromKeys(*values):
            cdef set[string] keys = values
            return Database.wrap(self.thisptr.only(keys))

        @default.register(int)
        def fromOffsets(*values):
            cdef set[ptrdiff_t] offsets = values
            return Database.wrap(self.thisptr.only(offsets))

        default.register(str, lambda *values: fromKeys(*[entry.encode() for entry in values]))
        return default(*entries)

    # Wraps an existing database instance.
    # @param target The database to be wrapped.
    # @return The new wrapper instance.
    @staticmethod
    cdef Database wrap(cDatabase& target):
        instance = <Database>Database.__new__(Database)
        instance.thisptr = target
        return instance

    # Informs the number of sequences in database.
    # @return int The number of sequences.
    @property
    def count(self):
        return self.thisptr.count()
