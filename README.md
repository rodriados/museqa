# Multiple Sequence Alignment
![license MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)
![version dev](https://img.shields.io/badge/version-dev-orange.svg)

Multiple Sequence Alignment using hybrid parallel computing. This project is initially based on
[@ferlete](https://github.com/ferlete)'s MSA project.

## Install

At the moment, this project is only being tested on Unix/Linux based systems. To install it, make sure you have both
`openmpi` and `cuda` installed in your system, download the source files and compile, following the steps below:

```
$ git clone https://github.com/rodriados/msa.git
$ make
```

## Usage

To use the project, you must execute via MPI. This can be achieved like the following:

```
$ mpirun -np <nproc> ./msa <file>
```

Where `nproc` is the number of processing nodes to use, being one of these the master node; and `file` being
the file containing the sequences to be aligned.