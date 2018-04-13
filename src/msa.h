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
#  define __debugh(...) fprintf(stderr, "[  msa:host] " __VA_ARGS__)
#  define __debugd(...) fprintf(stderr, "[msa:device] " __VA_ARGS__)
#else
#  define __debugh(...) if(verbose) {fprintf(stderr, "[  msa:host] " __VA_ARGS__);}
#  define __debugd(...) if(verbose) {fprintf(stderr, "[msa:device] " __VA_ARGS__);}
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

/** @struct sequence
 * @brief Represents a sequence to be processed.
 * @var length The sequence length.
 * @var data The sequence data.
 */
typedef struct {
    int length;
    char *data;
} sequence;

/** @struct msa_data
 * @brief Holds relevant data for common MSA functions.
 * @var scount The number of sequences given.
 * @var seq The list of sequences given.
 */
struct msa_data {
    int scount;
    sequence *seq;
};

/** @enum error_num
 * @brief Enumerates errors so their message can be easily printed.
 */
enum error_num {
    NOERROR = 0
,   NOFILE
,   INVALIDFILE
};

extern struct mpi_data mpi_data;
extern struct msa_data msa_data;
extern short verbose;

extern void finish(int);

#ifdef __cplusplus
}
#endif

#endif