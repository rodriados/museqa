/** @file msa.h
 * @brief Parallel Multiple Sequence Alignment main header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _MSA_H
#define _MSA_H

#define DEBUG
#define MSA "msa"
#define VERSION "0.1-alpha"

#include <stdio.h>

/* 
 * Checks whether the system we are compiling in is POSIX compatible. If it
 * is not POSIX compatible, some conditional compiling may take place.
 */
#if defined(unix) || defined(__unix__) || defined(__unix)                   \
    || defined(__linux__) || (defined(__APPLE__) && defined(__MACH__))
#  define __msa_posix__
#else
#  define __msa_noposix__
#endif

/*
 * Defines some debug macro functions. These functions should be used to
 * print out debugging information.
 */ 
#ifdef DEBUG
#  define __debugh(msg) fprintf(stderr, "[  msa:host] %s\n", msg)
#  define __debugd(msg) fprintf(stderr, "[msa:device] %s\n", msg)
#else
#  define __debugh(msg)
#  define __debugd(msg)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** @struct mpi_data
 * @brief Holds relevant data for MPI processing threads.
 * @var rank The current MPI thread rank.
 * @var nproc The total number of processing threads.
 */
struct mpi_data {
    int rank;
    int nproc;
};

/** @struct msa_data
 * @brief Holds relevant data for common MSA functions.
 */
struct msa_data {
};

extern struct mpi_data mpi_data;
extern struct msa_data msa_data;

extern void finish(int);

#ifdef __cplusplus
}
#endif

#endif