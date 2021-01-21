#!/usr/bin/env python
# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file Unit tests for the software's pairwise module.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2021-present Rodrigo Siqueira
from museqa.algorithm import pairwise as algorithm
from museqa.database import Database
from museqa import pairwise
import unittest

# Groups up a list of helper methods for tests for the pairwise module.
# @since 0.1.1
class TestPairwise(unittest.TestCase):
    # Defines some fixture files to parse and test against.
    # @since 0.1.1
    fixtures = [
        "fixtures/mock1.fasta"
    ,   "fixtures/mock2.fasta"
    ,   "fixtures/mock3.fasta"
    ]

    # Compares whether two distance matrices and asserts whether they're equal.
    # @param one The first distance matrix to be asserted.
    # @param two The second distance matrix to be asserted.
    def _compare(self, one, two):
        self.assertEqual(one.count, two.count)

        for i in range(one.count):
            for j in range(two.count):
                self.assertEqual(one[i, j], two[i, j])

    # Runs the pairwise module with different algorithms and compare their results.
    # @param tgt The target algorithm function to test the module with.
    # @param ref The reference function to compare the generated result with.
    # @param table An optional scoring table to use process sequences with.
    def _run(self, tgt, ref, table = None):
        for i, fixture in enumerate(TestPairwise.fixtures):
            with self.subTest(fixture = i):
                db = Database.load(fixture)

                expect = pairwise.run(db, table = table, algorithm = ref)
                result = pairwise.run(db, table = table, algorithm = tgt)

                self._compare(result, expect)

# Groups up a list of tests for the pairwise module's sequential algorithms.
# @since 0.1.1
class TestPairwiseSequential(TestPairwise):
    # Tests whether the sequential needleman algorithm returns the expected matrix.
    # @since 0.1.1
    def testNeedleman(self):
        self._run('sequential', algorithm.needleman)
        self._run('sequential', algorithm.needleman, table = 'blosum62')

# Groups up a list of tests for the pairwise module's hybrid algorithms.
# @since 0.1.1
class TestPairwiseHybrid(TestPairwise):
    # Tests whether the hybrid needleman algorithm returns the expected matrix.
    # @since 0.1.1
    @unittest.skipIf(True, "a CUDA device may not be available")
    def testNeedleman(self):
        self._run('hybrid', algorithm.needleman)
        self._run('hybrid', algorithm.needleman, table = 'blosum62')

# Defines the list of tests declared in the module. This is done manually for greater
# control on what is considered a test case or not.
cases = [
    TestPairwiseHybrid
,   TestPairwiseSequential
]

# Loads all test methods listed on the file from the list of test cases that must
# be present on the file.
# @since 0.1.1
def load_tests(loader, *_):
    tests = [loader.loadTestsFromTestCase(case) for case in cases]
    return unittest.TestSuite(tests)

if __name__ == '__main__':
    loader = unittest.TestLoader()
    unittest.main(testLoader = loader)
