/** @file fasta.h
 * @brief Parallel Multiple Sequence Alignment fasta header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _FASTA_H
#define _FASTA_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int allocfasta(const char *);
extern void freefasta();

#ifdef __cplusplus
}
#endif

#endif