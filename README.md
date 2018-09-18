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
To use the project, you must run it via MPI. A tool for finding all compatible hosts in your internal network is provided:
```bash
./findhosts
```
This will generate a file containing a recommended running configuration. To run it, you can simply use:
```bash
mpirun --hostfile hostfile ./msa <file>
```
Where `file` is the containing all sequences to be aligned.
