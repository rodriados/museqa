#!/usr/bin/env python
# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file Unit tests for the project's sequence module.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2019-present Rodrigo Siqueira
from museqa.sequence import Sequence, encode
import pytest

# Defines some fixture sequences to use for different test cases.
# @since 0.1.1
fixtures = [
    "MNNQRKKTGRPSFNMLKRARNRVSTGSQLAKRFSKGLLSGQGPMKLVMAFIAFLRFLAIPPTAGILARWS"
,   "TEVKGYTKGIDJAOIJDOADUIHDAIOAFAOIFJAOIFJOAIJFOAICJOAIMCOQIOIDUAITGYRQQ"
,   "MKNPKKKSGGFRIVNMLKRGVARVSPFGGLKRLPAGLLLGHGPIRMVLAILAFLRFTAIKPSLGLINRW"
,   "AAAAATCGATCAAGCAATCTCTCAGTCTCTCAGTTTACAGCAATCTTTGCAAACACACGAACCACAAA"
,   "LLNRFTMAHMKPTYERDVDLGAGTRHVAVEPEVANLDIIGQRIENIKNEHKSTWHYDEDNPYKTWAY"
]

# Creates a fixture for test sequences.
# @param request The test context to run the fixture with.
# @since 0.1.1
@pytest.fixture(params = fixtures)
def sequence(request):
    return map(lambda f: f(request.param), [Sequence, encode])

# Tests whether a sequence can be created and if they are created correctly.
# @param sequence The sequence to run the test with.
# @since 0.1.1
def testCanInstantiateSequence(sequence):
    target, expected = sequence
    assert expected == str(target)

# Tests whether one can access a random element in the sequence.
# @param sequence The sequence to run the test with.
# @since 0.1.1
def testCanAccessSequenceElement(sequence):
    target, expected = sequence

    for i in range(len(expected)):
        assert expected[i] == target[i]

# Tests whether sequence knows its correct length.
# @param sequence The sequence to run the test with.
# @since 0.1.1
def testIfSequenceHasCorrectLength(sequence):
    target, expected = sequence
    assert len(expected) == target.length

# Tests whether sequence raises correctly when accessing out of bounds.
# @param sequence The sequence to run the test with.
# @since 0.1.1
def testIfRaisesOnIndexError(sequence):
    target, _ = sequence

    with pytest.raises(IndexError):
        length = target.length
        ouch = target[length]
