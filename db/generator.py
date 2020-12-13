#!/usr/bin/env python
# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file A random sequence database generator for testing.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2019-present Rodrigo Siqueira
import random

# The list of available alphabets for sequence generation.
# @since 0.1.1
alphabet = {
    "dna": "ACTG"
,   "rna": "ACUG"
,   "protein": "ACTGRNDQEHILKMFPSWYVBJZ"
}

# Generates a random sequence of given length over given alphabet.
# @param alphabet The alphabet to generate sequence over.
# @param length The length of sequence to generate.
# @return The generated sequence of given length.
def generate(alphabet, length):
    letters = [random.choice(alphabet) for _ in range(length)]
    return str.join("", letters)

# Runs the generator with parameters given from command line.
# @param $1 The alphabet to use in sequences.
# @param $2 The number of sequences to create.
# @param $3 The length of each new sequence.
if __name__ == '__main__':
    from sys import argv

    # Generates a file name from given params.
    # @param count The number of sequences to create.
    # @param length The length of each sequence.
    # @return The name of file to create.
    def genfilename(count, length):
        return "generated%dxL%d.fasta" % (count, length)

    # Wraps a long sequence into multiple lines of given length.
    # @param sequence The sequence to be wrapped.
    # @param length The maximum line length.
    def wrap(sequence, length):
        return ["%s\n" % sequence[i:i + length] for i in range(0, len(sequence), length)]

    if len(argv) is not 4:
        print("Usage: python3 generator.py alphabet nsequences length")
        exit(1)

    chars = alphabet[argv[1]]
    count = int(argv[2])
    length = int(argv[3])

    with open(genfilename(count, length), "w") as file:
        for i in range(count):
            file.write(">anonymous#%d\n" % i)
            file.writelines(wrap(generate(chars, length), 70))
            file.write("\n")
