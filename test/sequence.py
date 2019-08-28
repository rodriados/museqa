from msa.sequence import Sequence
import unittest

class TestSequence(unittest.TestCase):

    def testCanCreate(self):
        self.assertEqual(str(Sequence("OI")), "XI*")
        self.assertEqual(str(Sequence("ACTGGGTCGATGCTACGTAC")), "ACTGGGTCGATGCTACGTAC*")

    def testCanAccessItens(self):
        original = "ACTGGGTCGATGCTACGTAC"
        sequence = Sequence(original)

        for i in range(len(original)):
            self.assertEqual(sequence[i], original[i])

    def testThrowOnIndexError(self):
        sequence = Sequence("ACTG")

        with self.assertRaises(RuntimeError) as context:
            a = sequence[10]

if __name__ == '__main__':
    unittest.main()
