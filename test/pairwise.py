#!/usr/bin/env python
# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file Unit tests for the project's pairwise module.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2021-present Rodrigo Siqueira
from museqa.algorithm import pairwise as algorithm
from museqa.database import Database
from museqa import pairwise
import pytest

# Defines some fixture files to be processed by different test cases.
# @since 0.1.1
fixtures = [
    "fixtures/mock1.fasta"
,   "fixtures/mock2.fasta"
,   "fixtures/mock3.fasta"
]

# Creates a fixture for sequence databases.
# @param request The test context to run the fixture with.
# @since 0.1.1
@pytest.fixture(params = fixtures)
def database(request):
    return Database.load(request.param)

# Creates a fixture for different scoring tables.
# @param request The test context to run the fixture with.
# @since 0.1.1
@pytest.fixture(params = ['default', 'blosum62', 'pam250'])
def table(request):
    return request.param

# Compares whether two distance matrices and asserts whether they're equal.
# @param expected The expected distance matrix to be asserted.
# @param produced The produced distance matrix to be asserted.
def assertTablesAreEqual(expected, produced):
    for i in range(expected.count):
        for j in range(produced.count):
            assert expected[i, j] == produced[i, j]

# Runs the pairwise module with different algorithms and compare their results.
# @param db The database to test the algorithms with.
# @param reference The reference function to compare the generated result with.
# @param target The target algorithm function to test the module with.
# @param extra Extra arguments to test the selected algorithm with.
def assertAlgorithmExecution(db, reference, target, **extra):
    expected = pairwise.run(db, algorithm = reference, **extra)
    produced = pairwise.run(db, algorithm = target, **extra)
    assert expected.count == produced.count
    
    assertTablesAreEqual(expected, produced)

# Tests whether the sequential needleman algorithm produces the expected matrix.
# @param database The database to test the algorithm with.
# @param table The scoring table to run the algorothm with.
# @since 0.1.1
def testSequentialNeedleman(database, table):
    assertAlgorithmExecution(database, 'sequential', algorithm.needleman, table = table)

# Tests whether the hybrid needleman algorithm produces the expected matrix.
# @param database The database to test the algorithm with.
# @param table The scoring table to run the algorothm with.
# @since 0.1.1
@pytest.mark.skip(reason = "a CUDA device may not be available")
def testHybridNeedleman(database, table):
    assertAlgorithmExecution(database, 'hybrid', algorithm.needleman, table = table)
