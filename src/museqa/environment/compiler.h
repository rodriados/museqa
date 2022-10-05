/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Compiler-specific configurations and macro definitions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

/*
 * Enumerating known host compilers. These compilers are not all necessarily officially
 * supported. Nevertheless, some special adaptation or fixes might be implemented
 * to each one of these if so needed.
 */
#define MUSEQA_HOST_COMPILER_UNKNOWN 0
#define MUSEQA_HOST_COMPILER_GCC     1

/*
 * Enumerating known device compilers. These compilers are not all necessarily officially
 * supported. Nevertheless, some special adaptation or fixes might be implemented
 * to each one of these if so needed.
 */
#define MUSEQA_DEVICE_COMPILER_UNKNOWN 0
#define MUSEQA_DEVICE_COMPILER_GCC     1
#define MUSEQA_DEVICE_COMPILER_NVCC    2

/*
 * Finds the version of the host compiler being used. Some features might change
 * or be unavailable depending on the current compiler configuration.
 */
#if defined(__GNUC__)
  #define MUSEQA_HOST_COMPILER MUSEQA_HOST_COMPILER_GCC
  #define MUSEQA_GCC_VERSION (__GNUC__ * 100 + __GNUC_MINOR__)
#else
  #define MUSEQA_HOST_COMPILER MUSEQA_HOST_COMPILER_UNKNOWN
#endif

/*
 * Finds the version of the device compiler being used. Some features might change
 * or be unavailable depending on the current compiler configuration.
 */
#if defined(__CUDACC__) || defined(__NVCOMPILER_CUDA__)
  #define MUSEQA_DEVICE_COMPILER MUSEQA_DEVICE_COMPILER_NVCC
#elif MUSEQA_HOST_COMPILER == MUSEQA_HOST_COMPILER_GCC
  #define MUSEQA_DEVICE_COMPILER MUSEQA_DEVICE_COMPILER_GCC
#else
  #define MUSEQA_DEVICE_COMPILER MUSEQA_DEVICE_COMPILER_UNKNOWN
#endif

/*
 * Macro for programmatically emitting a pragma call, independently on which compiler
 * is currently in use.
 */
#if !defined(MUSEQA_EMIT_PRAGMA_CALL)
  #define MUSEQA_EMIT_PRAGMA_CALL(x) _Pragma(#x)
#endif

/*
 * Macro for programmatically emitting a compiler warning. This will be shown during
 * compile-time and might stop compilation if warnings are treated as errors.
 */
#if !defined(MUSEQA_EMIT_COMPILER_WARNING)
  #if (MUSEQA_HOST_COMPILER == MUSEQA_HOST_COMPILER_GCC)
    #define MUSEQA_EMIT_COMPILER_WARNING(msg)             \
      MUSEQA_EMIT_PRAGMA_CALL(GCC warning msg)
  #else
    #define MUSEQA_EMIT_COMPILER_WARNING(msg)
  #endif
#endif

/*
 * Macro for disabling or manually emitting warnings when compiling host code with
 * GCC. This is useful for hiding buggy compiler warnings or compiler exploits that
 * have intentionally been taken advantage of.
 */
#if (MUSEQA_HOST_COMPILER == MUSEQA_HOST_COMPILER_GCC) && !defined(__CUDA_ARCH__)
  #define MUSEQA_EMIT_GCC_WARNING(x) MUSEQA_EMIT_COMPILER_WARNING(x)
  #define MUSEQA_DISABLE_GCC_WARNING_BEGIN(x)             \
    MUSEQA_EMIT_PRAGMA_CALL(GCC diagnostic push)          \
    MUSEQA_EMIT_PRAGMA_CALL(GCC diagnostic ignored x)
  #define MUSEQA_DISABLE_GCC_WARNING_END(x)               \
    MUSEQA_EMIT_PRAGMA_CALL(GCC diagnostic pop)
#else
  #define MUSEQA_DISABLE_GCC_WARNING_BEGIN(x)
  #define MUSEQA_DISABLE_GCC_WARNING_END(x)
  #define MUSEQA_EMIT_GCC_WARNING(x)
#endif

/*
 * Macro for disabling warnings when compiling device code with NVCC. This is useful
 * for hiding buggy compiler warnings or intentional compiler exploits.
 */
#if (MUSEQA_DEVICE_COMPILER == MUSEQA_DEVICE_COMPILER_NVCC)
  #define MUSEQA_DISABLE_NVCC_WARNING_BEGIN(x)            \
    MUSEQA_EMIT_PRAGMA_CALL(push)                         \
    MUSEQA_EMIT_PRAGMA_CALL(nv_diag_suppress = x)
  #define MUSEQA_DISABLE_NVCC_WARNING_END(x)              \
    MUSEQA_EMIT_PRAGMA_CALL(pop)
#else
  #define MUSEQA_DISABLE_NVCC_WARNING_BEGIN(x)
  #define MUSEQA_DISABLE_NVCC_WARNING_END(x)
#endif
