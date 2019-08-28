from msa.database import Database
from msa import parse, Sequence
import unittest

class TestDatabase(unittest.TestCase):

    def testCanInstantiate(self):
        parsed = parse("assets/mock.fasta")
        dbparsed = Database(parsed)
        dbadded = Database([Sequence("OLA"), Sequence("OI")])

        self.assertEqual(dbparsed.count, 10)
        self.assertEqual(dbadded.count, 2)

    def testCanAccessData(self):
        p0desc = 'AB492213.1 Hepatitis C virus subtype 1b gene for polyprotein, partial cds, strain: K4preS-20'
        p1desc = 'AB492212.1 Hepatitis C virus subtype 1b gene for polyprotein, partial cds, strain: K4preS-19'

        parsed = parse("assets/mock.fasta")
        dbparsed = Database(parsed)

        self.assertEqual(dbparsed[0].description, p0desc)
        self.assertEqual(dbparsed[1].description, p1desc)

if __name__ == '__main__':
    unittest.main()
