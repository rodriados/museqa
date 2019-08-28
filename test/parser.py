from msa import parse
import unittest

class TestParser(unittest.TestCase):

    def testParseFasta(self):
        p0desc = 'AB492213.1 Hepatitis C virus subtype 1b gene for polyprotein, partial cds, strain: K4preS-20'
        p1desc = 'AB492212.1 Hepatitis C virus subtype 1b gene for polyprotein, partial cds, strain: K4preS-19'

        p0data = [
                'CCCTGTGAGGAACTACTGTCTTCACGCAGAAAGCGTCTAGCCATGGCGTTAGTATGAGTGTCGTGCAGCC'
            ,   'TCCAGGACCCCCCCTCCCGGGAGAGCCATAGTGGTCTGCGGAACCGGTGAGTACACCGGAATTGCCAGGA'
            ,   'CGACCGGGTCCTTTCTTGGATTAACCCGCTCAATGCCTGGAGATTTGGGCGTGCCCCCGCGAGACTGCTA'
        ]

        p1data = [
                'CCCTGTGAGGAACTACTGTCTTCACGCAGAAAGCGTCTAGCCATGGCGTTAGTATGAGTGTCGTGCAGCC'
            ,   'TCCAGGACCCCCCCTCCCGGGAGAGCCATAGTGGTCTGCGGAACCGGTGAGTACACCGGAATTGCCAGGA'
            ,   'CGACCGGGTCCTTTCTTGGATTAACCCGCTCAATGCCTGGAGATTTGGGCGTGCCCCCGCGAGACTGCTA'
        ]

        parsed = parse("assets/mock.fasta")

        self.assertEqual(len(parsed), 10)
        self.assertEqual(len(str(parsed[0])), 210)

        self.assertEqual(parsed[0].description, p0desc)
        self.assertEqual(str(parsed[0]), "".join(p0data))

        self.assertEqual(parsed[1].description, p1desc)
        self.assertEqual(str(parsed[1]), "".join(p1data))

    def testThrowOnUnknownFile(self):
        with self.assertRaises(RuntimeError) as context:
            parse("assets/none.fasta")

    def testThrowOnUnknownExtension(self):
        with self.assertRaises(RuntimeError) as context:
            parse("assets/unknown.txt")

if __name__ == '__main__':
    unittest.main()
