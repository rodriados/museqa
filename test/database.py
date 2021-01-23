#!/usr/bin/env python
# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file Unit tests for the project's database module.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2019-present Rodrigo Siqueira
from museqa.database import Database
import pytest

# Defines a fixture database to use for different test cases.
# @since 0.1.1
fixtures = [
    {
        "SQ01": "MNNQRKKTGRPSFNMLKRARNRVSTGSQLAKRFSKGLLSGQGPMKLVMAFIAFLRFLAIPPTAGILA**"
    ,   "SQ02": "TEVKGYTKGXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX**"
    ,   "SQ03": "MKNPKKKSGGFRIVNMLKRGVARVSPFGGLKRLPAGLLLGHGPIRMVLAILAFLRFTAIKPSLGLIN**"
    ,   "SQ04": "AAAAATCGATCAAGCAATCTCTCAGTCTCTCAGTTTACAGCAATCTTTGCAAACACACGAACCACAA**"
    ,   "SQ05": "LLNRFTMAHMKPTYERDVDLGAGTRHVAVEPEVANLDIIGQRIENIKNEHKSTWHYDEDNPYKTWAY**"
    }
,   {
        "SQ06": "GATAAAAGAACCTATAATCCCTTCGCACACCGCGTCACACCGCGCTATATGCTGCTCATTAGGAAT"
    ,   "SQ07": "GAGCAATCGTTACATAACATGCATTCCAAATAAGGGAATGCGGAAATCCATACTTGGGACTTACAT"
    ,   "SQ08": "GAGCAATCGTTACATAACATGCATTCTAACTAAGGCAATGCGGGATTCCATACTTGGGACTTACAT"
    ,   "SQ09": "GCTCCTTTTTTGTGGATACAATCTCTTGTATACGATATACTTATTGTTAATTTCATTGACCTTTAC"
    ,   "SQ10": "TGCATTCCATATAAGGAATACGGATATGCCCATGTTTGTATCCAAACAGGCGGTCTCCCAGACTCC"
    }
]

# Creates a fixture for databases
# @param request The test context to run the fixture with.
# @since 0.1.1
@pytest.fixture(params = fixtures)
def database(request):
    return request.param

# Creates a fixture for split databases.
# @param request The test context to run the fixture with.
# @param database The database sequences to test with.
# @since 0.1.1
@pytest.fixture(params = [2, 3, 4])
def subsets(request, database):
    items = [*database.items()]
    return items[:request.param], items[request.param:]

# Tests whether sequences can be added to the database.
# @param database The database sequences to test with.
# @since 0.1.1
def testIfCanAddSequences(database):
    db = Database()
    db.add(database)

    for name, sequence in database.items():
        assert name == db[name].description
        assert sequence == str(db[name].contents)

    assert len(database) == db.count

# Tests whether two databases can be merged into a single one.
# @since 0.1.1
def testIfCanMergeDatabases(subsets):
    db = []

    for i, subset in enumerate(subsets):
        db.append(Database())
        db[i].add(subset)
        assert len(subset) == db[i].count

    for i in range(1, len(db)):
        db[0].merge(db[i])

    for subset in subsets:
        for name, sequence in subset:
            assert name == db[0][name].description
            assert sequence == str(db[0][name].contents)

    assert sum(len(s) for s in subsets) == db[0].count
