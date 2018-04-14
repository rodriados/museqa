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

/** @enum errornum_t
 * @brief Enumerates errors so their message can be easily printed.
 */
typedef enum {
    NOERROR = 0
,   NOFILE
,   INVALIDFILE
} errornum_t;

/** @struct mpidata_t
 * @brief Holds relevant data for MPI processing threads.
 * @var rank The current MPI thread rank.
 * @var nproc The total number of processing threads.
 */
typedef struct {
    int rank;
    int nproc;
} mpidata_t;

/** @struct sequence_t
 * @brief Represents a sequence to be processed.
 * @var length The sequence length.
 * @var data The sequence data.
 */
typedef struct {
    unsigned int length;
    char *data;
} sequence_t;

/**
 * @struct workpair_t
 * @brief Holds a pair of sequences to be aligned.
 * @var seq The pair of sequences to align.
 */
typedef struct {
    short seq[2];
} workpair_t;

/** @struct msadata_t
 * @brief Holds relevant data for common MSA functions.
 * @var npair The number of given work-pairs.
 * @var nseq The number of given sequences.
 * @var pair The list of given work-pairs.
 * @var seq The list of given sequences.
 */
typedef struct {
    int npair;
    short nseq;
    workpair_t *pair;
    sequence_t *seq;
} msadata_t;

extern short verbose;
extern void finish(errornum_t);

#ifdef __cplusplus
}
#endif

#endif