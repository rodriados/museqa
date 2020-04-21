#!/usr/bin/env python
# Multiple Sequence Alignment database test file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2019-2020 Rodrigo Siqueira
from msa.database import Database
from msa.sequence import Sequence
from msa import parse
import unittest

# Groups up a list of tests for database objects.
# @since 0.1.1
class TestDatabase(unittest.TestCase):
    # Defines some fixture entries to test against.
    # @since 0.1.1
    fixtures = {
        "SQ01": "MNNQRKKTGRPSFNMLKRARNRVSTGSQLAKRFSKGLLSGQGPMKLVMAFIAFLRFLAIPPTAGILA**"
    ,   "SQ02": "TEVKGYTKGXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX**"
    ,   "SQ03": "MKNPKKKSGGFRIVNMLKRGVARVSPFGGLKRLPAGLLLGHGPIRMVLAILAFLRFTAIKPSLGLIN**"
    ,   "SQ04": "AAAAATCGATCAAGCAATCTCTCAGTCTCTCAGTTTACAGCAATCTTTGCAAACACACGAACCACAA**"
    ,   "SQ05": "LLNRFTMAHMKPTYERDVDLGAGTRHVAVEPEVANLDIIGQRIENIKNEHKSTWHYDEDNPYKTWAY**"
    }

    # Compares entries to fixtures. This allows us to test different type constructors.
    # @param entries The database entries to compare fixtures with.
    # @since 0.1.1
    def compare(self, entries):
        db = Database()
        db.add(entries)

        for fixture, contents in TestDatabase.fixtures.items():
            self.assertEqual(fixture, db[fixture].description)
            self.assertEqual(contents, str(db[fixture].contents))

        self.assertEqual(len(TestDatabase.fixtures), db.count)

    # Tests whether sequences can be added to the database.
    # @since 0.1.1
    def add(self):
        self.compare(TestDatabase.fixtures)
        self.compare([(key, val) for key, val in TestDatabase.fixtures.items()])
        self.compare([(key, Sequence(val)) for key, val in TestDatabase.fixtures.items()])

    # Tests whether sequences can be accessed via index and description.
    # @since 0.1.1
    def access(self):
        db = Database()
        db.add(TestDatabase.fixtures)

        for i, fixture in enumerate(TestDatabase.fixtures):
            self.assertEqual(fixture, db[i].description)
            self.assertEqual(fixture, db[fixture].description)

    # Tests whether two databases can be merged into a single one.
    # @since 0.1.1
    def merge(self):
        db = [Database(), Database()]
        subset = [["SQ01", "SQ03"], ["SQ04", "SQ02", "SQ05"]]

        db[0].add({key: val for key, val in TestDatabase.fixtures.items() if key in subset[0]})
        db[1].add({key: val for key, val in TestDatabase.fixtures.items() if key in subset[1]})

        self.assertEqual(len(subset[0]), db[0].count)
        self.assertEqual(len(subset[1]), db[1].count)
        db[0].merge(db[1])

        for fixture, contents in TestDatabase.fixtures.items():
            self.assertEqual(fixture, db[0][fixture].description)
            self.assertEqual(contents, str(db[0][fixture].contents))

    testCanAddSequences = add
    testSequencesAccess = access
    testCanMerge = merge

if __name__ == '__main__':
    unittest.main()
