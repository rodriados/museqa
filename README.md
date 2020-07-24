# Multiple Sequence Alignment
![license MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)
![version dev](https://img.shields.io/badge/version-dev-orange.svg)

Multiple Sequence Alignment using hybrid parallel computing.

## Motivation
In bioinformatics, the global alignment of multiple biological sequences has crucial importance. This allows
biologists to understand, among many other things, evolutionary relations between different beings and species. However,
the optimal solution to this problem is computationally difficult, that is, it is not executable in a timely manner
if the number of sequences or the size of such sequences are arbitrarily large. As alternatives, several approaches
have arisen as heuristics to this problem.

This project is focused in exploring and implementing the Progressive Alignment heuristics using hybrid parallel
computing in all of its three separate steps: pairwise alignment; phylogenetic tree generation; and finally the progressive
alignment itself.

The time complexity of this heuristic is O(_n_<sup>2</sup>_l_<sup>2</sup>+_n_<sup>3</sup>) to align _n_ sequences of
average size _l_. Note that, even using this heuristic, time constraints are still prohibitive. For this reason, this
project uses GPGPU parallelism to attack the heuristic's execution time.

## Install
Up to this moment, only has this project been targeted to and tested in Unix/Linux environments and NVIDIA CUDA-enabled
GPGPUs. To install it, make sure you have both `openmpi` and `cuda` installed in your system, download the source files and compile, following the steps below:
```bash
git clone https://github.com/rodriados/msa.git
make
```

Optionally, this project can also be exported to Python, primarily for testing purposes. For such, you will also need `cython`
installed in your system and compile with:
```bash
make testing
```

## Usage
To use this project, you simply run:
```bash
msarun <file>
```
Where `file` is the file containing all sequences to be aligned.
