#!/usr/bin/env python
# Multiple Sequence Alignment random sequence database generator file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2019 Rodrigo Siqueira
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
    return "".join([random.choice(alphabet) for i in range(length)])

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
        return "".join(["generated", str(count) , "x", str(length), "l.fasta"])

    # Wraps a long string into multiple lines of given length.
    # @param string The string to be wrapped.
    # @param linelen The maximum line length.
    def wrap(string, linelen):
        return [string[i : i + linelen] + "\n" for i in range(0, len(string), linelen)]

    chars = alphabet[argv[1]]
    count = int(argv[2])
    length = int(argv[3])

    with open(genfilename(count, length), "w") as file:
        for i in range(count):
            file.write(">" + str(i) + "\n")
            file.writelines(wrap(generate(chars, length), 70))
            file.write("\n")
