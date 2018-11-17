/** 
 * Multiple Sequence Alignment helper functions header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef HELPER_H_INCLUDED
#define HELPER_H_INCLUDED

#pragma once

#include <stdio.h>
#include <stdint.h>

#include "colors.h"

/*
 * Defines some info macro functions. These functions should be used to
 * print out information.
 */ 
#ifndef __CUDA_ARCH__
#define info(msg, ...) fprintf(stderr, s_bold "[   info]: " s_reset msg "\n", ##__VA_ARGS__)
#else
#define info(msg, ...) // Please, do not print anything from device
#endif

#ifdef __cplusplus
extern "C" {
#endif
/**
 * This struct conveys error information so they can be easily created, returned
 * by functions and accessed anywhere.
 * @since 0.1.alpha
 */
typedef struct
{
    const char *msg = NULL;             /// The error message. Use NULL for success.
    alignas(intptr_t) int8_t code = 0;  /// The error code.
} Error;

/*
 * Declaring global functions.
 */
extern void usage();
extern void version();
extern void finalize(Error);

#ifdef __cplusplus
}
#endif

#endif