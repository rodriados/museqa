# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file The project's docker container configuration.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2021-present Rodrigo Siqueira

# This container is based on Nvidia's CUDA container.
# This version is the one used during development.
FROM nvidia/cuda:10.2-devel-ubuntu18.04

LABEL maintainer="rodriados@gmail.com"

ENV USER museqa

ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/home/${USER}

# Installing all project's dependencies such as MPI and Cython.
# These are used to compile and run the project.
RUN apt-get update -y
RUN apt-get install -y --no-install-recommends \
       sudo apt-utils openssh-server gcc libopenmpi-dev openmpi-bin openmpi-common \
       binutils python3 python3-dev python3-pip make
RUN pip3 install cython pytest

RUN apt-get clean && apt-get purge
RUN rm -rf /var/lib/apt/lists/*

# Compiles the project. The docker container is created with the project's code
# already compiled, so it's easier to use.
WORKDIR /museqa
COPY . /museqa

RUN ["make", "clean"]
RUN ["make", "production", "-j4"]
RUN ["make", "testing", "-j4"]
