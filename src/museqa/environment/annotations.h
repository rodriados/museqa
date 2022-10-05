/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Definition of compiler-specific annotations.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment/compiler.h>

/*
 * Since only NVCC knows how to deal with `__host__` and `__device__` annotations,
 * we define them to empty strings when another compiler is in use. This allows
 * the use of these annotations without needing to care whether they'll be known
 * by the compiler or not.
 */
#if MUSEQA_DEVICE_COMPILER != MUSEQA_DEVICE_COMPILER_NVCC
  #if !defined(__host__)
    #define __host__
  #endif

  #if !defined(__device__)
    #define __device__
  #endif
#endif

/*
 * Defining the __forceinline__ annotation for all compilers, allowing code to be
 * seamlessly compiled independently of the host or device compilers in use.
 */
#if !defined(__CUDACC__) && !defined(__NVCOMPILER_CUDA__)
  #if !defined(__forceinline__)
    #define __forceinline__
  #endif
#endif
