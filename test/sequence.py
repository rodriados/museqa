#!/usr/bin/env python
# Multiple Sequence Alignment sequence test file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2019-2020 Rodrigo Siqueira
from msa.sequence import Sequence
import unittest

# The list of available alphabets for sequences generation.
# @since 0.1.1
alphabet = "ACTGRNDQEHILKMFPSWYVBJUZ"

# The character to use as a placeholder when an invalid sequence character is found.
# @since 0.1.1
invalid  = "X"

# The character to indicate padding after a sequence has already been finished.
# @since 0.1.1
padding  = "*"

# Implements the encoding algorithm for testing. We prefer reimplementing the
# algorithm, instead of simply showing the expected result to avoid using the
# original algorithm's outputs as correct result.
# @param sequence The sequence to be encoded.
# @return The encoded sequence.
def encode(sequence):
    length = len(sequence)
    result = []

    for letter in sequence:
        result.append(letter if letter in alphabet else invalid)

    if length % 3 is not 0:
        result += padding * (3 - (length % 3))

    return str.join("", result)

# Groups up a list of tests for sequence objects.
# @since 0.1.1
class TestSequence(unittest.TestCase):
    # Defines some fixtures cases for different test cases.
    # @since 0.1.1
    fixtures = [
        "MNNQRKKTGRPSFNMLKRARNRVSTGSQLAKRFSKGLLSGQGPMKLVMAFIAFLRFLAIPPTAGILARWS"
    ,   "TEVKGYTKGXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    ,   "MKNPKKKSGGFRIVNMLKRGVARVSPFGGLKRLPAGLLLGHGPIRMVLAILAFLRFTAIKPSLGLINRW"
    ,   "AAAAATCGATCAAGCAATCTCTCAGTCTCTCAGTTTACAGCAATCTTTGCAAACACACGAACCACAAA"
    ,   "LLNRFTMAHMKPTYERDVDLGAGTRHVAVEPEVANLDIIGQRIENIKNEHKSTWHYDEDNPYKTWAY"
    ]

    # Tests whether a sequence can be created and if they are created correctly.
    # @since 0.1.1
    def instantiate(self):
        for fixture in TestSequence.fixtures:
            expected = encode(fixture)
            sequence = str(Sequence(fixture))

            self.assertEqual(sequence, expected)

    # Tests whether one can access a random element in the sequence.
    # @since 0.1.1
    def access(self):
        for fixture in TestSequence.fixtures:
            expected = encode(fixture)
            sequence = Sequence(fixture)

            for i in range(len(expected)):
                self.assertEqual(sequence[i], expected[i])

    # Tests whether sequence knows its correct length.
    # @since 0.1.1
    def length(self):
        for fixture in TestSequence.fixtures:
            expected = encode(fixture)
            sequence = Sequence(fixture)

            self.assertEqual(sequence.length, len(expected))

    # Tests whether sequence raises correctly when accessing out of bounds.
    # @since 0.1.1
    def indexError(self):
        for fixture in TestSequence.fixtures:
            sequence = Sequence(fixture)

            with self.assertRaises(IndexError) as context:
                length = sequence.length
                ouch = sequence[length]

    testCanInstantiate = instantiate
    testCanAccessItem = access
    testKnowsLength = length
    testThrowIndexError = indexError

if __name__ == '__main__':
    unittest.main()
