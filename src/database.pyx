#!/usr/bin/env python
# cython: language_level = 2
# Multiple Sequence Alignment sequence database wrapper file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2019 Rodrigo Siqueira
from libcpp.set cimport set
from libcpp.string cimport string
from sequence cimport cSequence, Sequence
from database cimport cDatabase, cDatabaseEntry

# Represents a sequence stored in database.
# @since 0.1.1
cdef class DatabaseEntry:

    # Initializes a new database entry representation.
    # @param string description The sequence description
    # @param string sequence The sequence store in database
    def __cinit__(self, string description, string sequence):
        self.cRef.description = description
        self.cRef.sequence = cSequence(sequence)

    # Transforms the entry into the sequence for exhibition.
    # @return The database entry representation as a string.
    def __str__(self):
        return self.cRef.sequence.toString()

    @property
    # Gives access to the entry description string.
    # @return The entry description.
    def description(self):
        return self.cRef.description

    @property
    # Gives access to the entry sequence.
    # @return The entry sequence.
    def sequence(self):
        return Sequence(self.cRef.sequence.toString())

# Stores a list of sequences read from possible different sources. The
# sequences may be identified by description or inclusion index.
# @since 0.1.1
cdef class Database:

    # Instantiates a new sequence database.
    # @param list args Positional arguments.
    def __cinit__(self, *args):
        self.add(*args)

    # Gives access to a specific sequence of the database.
    # @param int offset The requested sequence offset.
    # @return Sequence The requested sequence.
    def __getitem__(self, int offset):
        cdef cDatabaseEntry entry = self.cRef.getEntry(offset)
        return DatabaseEntry(entry.description, entry.sequence.toString())

    # Adds new sequence(s) to database.
    # @param mixed arg The sequence(s) to add.
    def add(self, *args):
        for arg in args:
            if isinstance(arg, list):
                self.add(*arg)
            elif isinstance(arg, dict):
                self.__addFromDict(arg)
            elif isinstance(arg, tuple):
                self.__addFromTuple(arg[0], arg[1])
            elif isinstance(arg, str):
                self.__addFromString(arg)
            elif isinstance(arg, Sequence):
                self.__addFromSequence(arg)
            elif isinstance(arg, Database):
                self.__addFromDatabase(arg)
            elif isinstance(arg, DatabaseEntry):
                self.__addFromDatabaseEntry(arg)
            else:
                raise ValueError("Unknown sequence type.")

    # Removes sequence(s) from database.
    # @param int offsets The sequence(s) offset(s) to remove.
    def remove(self, *offsets):
        cdef set[ptrdiff_t] indeces = [int(i) for i in offsets]
        self.cRef.removeMany(indeces)

    @property
    # Informs the number of sequences in database.
    # @return int The number of sequences.
    def count(self):
        return self.cRef.getCount()

    # Adds new database entries from a key-value dict.
    # @param dict arg The dict to be added to database.
    def __addFromDict(self, dict arg):
        for key, value in arg.iteritems():
            self.__addFromTuple(str(key), value)

    # Adds new database entries from an already existing database.
    # @param dbase The database to be fused.
    cdef void __addFromDatabase(self, Database dbase):
        self.cRef.addMany(dbase.cRef)

    # Adds a new database entry from an already existing database entry.
    # @param entry The entry to be added to database.
    cdef void __addFromDatabaseEntry(self, DatabaseEntry entry):
        self.cRef.add(entry.cRef)

    # Adds a new database entry from an already existing sequence.
    # @param sequence The sequence to be added to database.
    cdef void __addFromSequence(self, Sequence sequence):
        self.cRef.add(sequence.cRef)

    # Adds a new database entry from a sequence as string.
    # @param sequence The sequence to be added to database.
    cdef void __addFromString(self, string sequence):
        self.cRef.add(cSequence(sequence))

    # Adds a new database entry from a entry tuple.
    # @param desc The sequence description.
    # @param sequence The sequence to be added to database.
    cdef void __addFromTuple(self, string desc, string sequence):
        self.cRef.add(desc, cSequence(sequence))
