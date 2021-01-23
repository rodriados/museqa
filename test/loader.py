#!/usr/bin/env python
# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file Unit tests for the project's IO loader module.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2019-present Rodrigo Siqueira
from museqa.database import Database
from hashlib import md5
import pytest

# Defines some fixture files alongside their expected contents to parse and test
# against. The files listed on this dictionary are expected to exist inside a folder
# on the root test directory.
# @since 0.1.1
fixtures = {
    'fasta': {
        "fixtures/mock1.fasta": {
            "NC_007728.1": "50e1a4367bd08aa24b96dec226e54b61"
        ,   "AB492213.1":  "40def109d76a2e7ad42b4c6554cf2843"
        ,   "AY186732.1":  "1aafda82b3084e002ee47ec466c648fb"
        ,   "AY590471.1":  "b97c3f6adf77956a6111b67380ed2433"
        ,   "NC_032975.1": "7bdc5d496aed747991b11d0d182b0eed"
        }
    ,   "fixtures/mock2.fasta": {
            "AF241168.1":  "c638e233669d5a37e88eacfad7e3d435"
        ,   "NC_011560.1": "19cf53404d9129ade12e7d97d82f2b8a"
        ,   "NC_001874.1": "95bd6fb267dcab96ea781e90dcabfbe4"
        }
    ,   "fixtures/mock3.fasta": {
            "HQ541801.1":  "81c8ea16a4fdf2d13ecb69048a98bd1b"
        ,   "L03295.1":    "6f4603a0d141561f2d73dcbf71b529ea"
        ,   "KY785481.1":  "d198d09fdd802b5ddae606b569299fda"
        ,   "KY325470.1":  "aba55fc4c6a7604cc7329bd7b411f00f"
        ,   "NC_032632.1": "e25645acbbcd44f62088e366afa13984"
        }
    }
}

# Creates a fixture for FASTA files.
# @param request The test context to run the fixture with.
# @since 0.1.1
@pytest.fixture(params = dict.items(fixtures['fasta']))
def fasta(request):
    return request.param

# Tests whether FASTA files can be parsed correctly.
# @param fasta The file fixture to run the test with.
# @since 0.1.1
def testLoadFromFastaFile(fasta):
    filename, expected = fasta
    database = Database.load(filename)

    for key, digest in expected.items():
        assert key == database[key].description
        assert digest == md5(str(database[key].contents).encode()).hexdigest()

# Tests whether an exception is thrown when trying to parse unknown file.
# @since 0.1.1
def testIfRaisesOnUnknownFile():
    with pytest.raises(RuntimeError):
        Database.load('unknown/file/path')
