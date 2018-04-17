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
#  define __msaposix__
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
,   INVALIDARG
,   NOGPUFOUND
,   CUDAERROR
} errornum_t;

/** @struct mpidata_t
 * @brief Holds relevant data for MPI processing threads.
 * @var rank The current MPI thread rank.
 * @var size The total number of processing threads.
 */
typedef struct {
    int rank;
    int size;
} mpidata_t;

/** @struct sequence_t
 * @brief Represents a sequence to be processed.
 * @var length The sequence length.
 * @var data The sequence data.
 */
typedef struct {
    char *data;
    unsigned length;
} sequence_t;

extern mpidata_t mpi_data;
extern unsigned char verbose;

extern void finish(errornum_t);

#ifdef __cplusplus
}
#endif

/*
 * Defines some debug macro functions. These functions should be used to
 * print out debugging information.
 */ 
#ifdef DEBUG
#  define __debugh(msg, ...) fprintf(stderr, "[  msa:host] " msg "\n", ##__VA_ARGS__)
#  define __debugd(msg, ...) printf("[msa:device] " msg "\n", ##__VA_ARGS__)
#else
#  define __debugh(msg, ...) if(verbose) {fprintf(stderr, "[  msa:host] " msg "\n", ##__VA_ARGS__);}
#  define __debugd(msg, ...) if(verbose) {printf("[msa:device] " msg "\n", ##__VA_ARGS__);}
#endif

/*
 * Defines some process control macros. These macros are to be used when
 * it is needed to check whether the current process is master or not.
 */
#define __master 0
#define __ismaster() (mpi_data.rank == __master)
#define __isslave()  (mpi_data.rank != __master)
#define __onlymaster   if(__ismaster())
#define __onlyslaves   if(__isslave())
#define __onlyslave(i) if(__isslave() && mpi_data.rank == i)

#endif