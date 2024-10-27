/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Environment configuration and macro values.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/version.h>

/*
 * Enumerates all possible target environment modes to which the code might be compiled
 * to. The environment mode may affect some features' availability and performace.
 */
#define MUSEQA_BUILD_DEV         0
#define MUSEQA_BUILD_DEBUG       1
#define MUSEQA_BUILD_TESTING     2
#define MUSEQA_BUILD_PRODUCTION  3
#define MUSEQA_BUILD_PERFORMANCE 9

/*
 * Discovers and explicits the target environment mode to which the code must be
 * currently compiled to. The mode may affect some features' availability and performance.
 */
#if defined(DEBUG) || defined(_DEBUG)
  #define MUSEQA_BUILD MUSEQA_BUILD_DEBUG
  #define MUSEQA_ENVIRONMENT "Debug"
#elif defined(TESTING)
  #define MUSEQA_BUILD MUSEQA_BUILD_TESTING
  #define MUSEQA_ENVIRONMENT "Testing"
#elif defined(DEV) || defined(DEVELOPMENT)
  #define MUSEQA_BUILD MUSEQA_BUILD_DEV
  #define MUSEQA_ENVIRONMENT "Development"
#elif defined(MUSEQA_PERFORMANCE)
  #define MUSEQA_BUILD MUSEQA_BUILD_PERFORMANCE
  #define MUSEQA_ENVIRONMENT "Performance"
#else
  #define MUSEQA_BUILD MUSEQA_BUILD_PRODUCTION
  #define MUSEQA_ENVIRONMENT "Production"
#endif

/*
 * Enumerating known host compilers. These compilers are not all necessarily officially
 * supported. Nevertheless, some special adaptation or fixes might be implemented
 * to each one of these if so needed.
 */
#define MUSEQA_OPT_HOST_COMPILER_UNKNOWN 0
#define MUSEQA_OPT_HOST_COMPILER_GCC     1
#define MUSEQA_OPT_HOST_COMPILER_CLANG   2
#define MUSEQA_OPT_HOST_COMPILER_NVCC    3

/*
 * Enumerating known device compilers. These compilers are not all necessarily officially
 * supported. Nevertheless, some special adaptation or fixes might be implemented
 * to each one of these if so needed.
 */
#define MUSEQA_OPT_DEVICE_COMPILER_UNKNOWN 0
#define MUSEQA_OPT_DEVICE_COMPILER_CLANG   2
#define MUSEQA_OPT_DEVICE_COMPILER_NVCC    3

/*
 * Finds the version of the host compiler being used. Some features might change
 * or be unavailable depending on the current compiler configuration.
 */
#if defined(__clang__)
  #define MUSEQA_HOST_COMPILER MUSEQA_OPT_HOST_COMPILER_CLANG
  #define MUSEQA_CLANG_VERSION (__clang_major__ * 100 + __clang_minor__)
  #define MUSEQA_HOST_COMPILER_VERSION MUSEQA_CLANG_VERSION

#elif defined(__NVCC__)
  #define MUSEQA_HOST_COMPILER MUSEQA_OPT_HOST_COMPILER_NVCC
  #define MUSEQA_NVCC_VERSION (__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__)
  #define MUSEQA_HOST_COMPILER_VERSION MUSEQA_NVCC_VERSION

#elif defined(__GNUC__)
  #define MUSEQA_HOST_COMPILER MUSEQA_OPT_HOST_COMPILER_GCC
  #define MUSEQA_GCC_VERSION (__GNUC__ * 100 + __GNUC_MINOR__)
  #define MUSEQA_HOST_COMPILER_VERSION MUSEQA_GCC_VERSION

#else
  #define MUSEQA_HOST_COMPILER MUSEQA_OPT_HOST_COMPILER_UNKNOWN
  #define MUSEQA_HOST_COMPILER_VERSION 0
#endif

/*
 * Finds the version of the device compiler being used. Some features might change
 * or be unavailable depending on the current compiler configuration.
 */
#if defined(__CUDACC__) && defined(MUSEQA_NVCC_VERSION)
  #define MUSEQA_DEVICE_COMPILER MUSEQA_OPT_DEVICE_COMPILER_NVCC
  #define MUSEQA_DEVICE_COMPILER_VERSION MUSEQA_NVCC_VERSION

#elif defined(__CUDACC__) && defined(MUSEQA_CLANG_VERSION)
  #define MUSEQA_DEVICE_COMPILER MUSEQA_OPT_DEVICE_COMPILER_CLANG
  #define MUSEQA_DEVICE_COMPILER_VERSION MUSEQA_CLANG_VERSION

#else
  #define MUSEQA_DEVICE_COMPILER MUSEQA_OPT_DEVICE_COMPILER_UNKNOWN
  #define MUSEQA_DEVICE_COMPILER_VERSION 0
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
  #if (MUSEQA_HOST_COMPILER == MUSEQA_OPT_HOST_COMPILER_GCC \
    || (MUSEQA_HOST_COMPILER == MUSEQA_OPT_HOST_COMPILER_NVCC) && !defined(__CUDA_ARCH__))
    #define MUSEQA_EMIT_COMPILER_WARNING(msg) \
      MUSEQA_EMIT_PRAGMA_CALL(GCC warning msg)

  #elif (MUSEQA_HOST_COMPILER == MUSEQA_OPT_HOST_COMPILER_CLANG)
    #define MUSEQA_EMIT_COMPILER_WARNING(msg) \
      MUSEQA_EMIT_PRAGMA_CALL(clang warning msg)

  #else
    #define MUSEQA_EMIT_COMPILER_WARNING(msg)
  #endif
#endif

/*
 * Macros for disabling or manually emitting warnings with specific compilers. This
 * is useful to treat the behaviour of a specific compiler, such as hiding buggy
 * compiler warnings or exploits that have intentionally been taken advantage of.
 */
#if (MUSEQA_HOST_COMPILER == MUSEQA_OPT_HOST_COMPILER_GCC)
  #define MUSEQA_EMIT_GCC_WARNING(x) MUSEQA_EMIT_COMPILER_WARNING(x)
  #define MUSEQA_DISABLE_GCC_WARNING_BEGIN(x)             \
    MUSEQA_EMIT_PRAGMA_CALL(GCC diagnostic push)          \
    MUSEQA_EMIT_PRAGMA_CALL(GCC diagnostic ignored x)
  #define MUSEQA_DISABLE_GCC_WARNING_END(x)               \
    MUSEQA_EMIT_PRAGMA_CALL(GCC diagnostic pop)
  #define MUSEQA_PUSH_GCC_OPTION_BEGIN(x)                 \
    MUSEQA_EMIT_PRAGMA_CALL(GCC push_options)             \
    MUSEQA_EMIT_PRAGMA_CALL(GCC x)
  #define MUSEQA_PUSH_GCC_OPTION_END(x)                   \
    MUSEQA_EMIT_PRAGMA_CALL(GCC pop_options)
#else
  #define MUSEQA_EMIT_GCC_WARNING(x)
  #define MUSEQA_DISABLE_GCC_WARNING_BEGIN(x)
  #define MUSEQA_DISABLE_GCC_WARNING_END(x)
  #define MUSEQA_PUSH_GCC_OPTION_BEGIN(x)
  #define MUSEQA_PUSH_GCC_OPTION_END(x)
#endif

#if (MUSEQA_HOST_COMPILER == MUSEQA_OPT_HOST_COMPILER_CLANG)
  #define MUSEQA_EMIT_CLANG_WARNING(x) MUSEQA_EMIT_COMPILER_WARNING(x)
  #define MUSEQA_DISABLE_CLANG_WARNING_BEGIN(x)           \
    MUSEQA_EMIT_PRAGMA_CALL(clang diagnostic push)        \
    MUSEQA_EMIT_PRAGMA_CALL(clang diagnostic ignored x)
  #define MUSEQA_DISABLE_CLANG_WARNING_END(x)             \
    MUSEQA_EMIT_PRAGMA_CALL(clang diagnostic pop)
  #define MUSEQA_PUSH_CLANG_ATTRIBUTE_BEGIN(x)            \
    MUSEQA_EMIT_PRAGMA_CALL(clang attribute push (x))
  #define MUSEQA_PUSH_CLANG_ATTRIBUTE_END                 \
    MUSEQA_EMIT_PRAGMA_CALL(clang attribute pop)
#else
  #define MUSEQA_EMIT_CLANG_WARNING(x)
  #define MUSEQA_DISABLE_CLANG_WARNING_BEGIN(x)
  #define MUSEQA_DISABLE_CLANG_WARNING_END(x)
  #define MUSEQA_PUSH_CLANG_ATTRIBUTE_BEGIN(x)
  #define MUSEQA_PUSH_CLANG_ATTRIBUTE_END
#endif

#if (MUSEQA_HOST_COMPILER == MUSEQA_OPT_HOST_COMPILER_NVCC)
  #if !defined(__CUDA_ARCH__)
    #define MUSEQA_EMIT_NVCC_WARNING(x) MUSEQA_EMIT_COMPILER_WARNING(x)
  #else
    #define MUSEQA_EMIT_NVCC_WARNING(x)
  #endif
  #define MUSEQA_DISABLE_NVCC_WARNING_BEGIN(x)            \
    MUSEQA_EMIT_PRAGMA_CALL(push)                         \
    MUSEQA_EMIT_PRAGMA_CALL(diag_suppress = x)
  #define MUSEQA_DISABLE_NVCC_WARNING_END(x)              \
    MUSEQA_EMIT_PRAGMA_CALL(diag_default = x)
#else
  #define MUSEQA_EMIT_NVCC_WARNING(x)
  #define MUSEQA_DISABLE_NVCC_WARNING_BEGIN(x)
  #define MUSEQA_DISABLE_NVCC_WARNING_END(x)
#endif

/*
 * Discovers the C++ language dialect in use for the current compilation. A specific
 * dialect might not be supported or might be required for certain functionalities
 * to work properly.
 */
#if defined(__cplusplus)
  #if __cplusplus < 201103L
    #define MUSEQA_CPP_DIALECT 2003
  #elif __cplusplus < 201402L
    #define MUSEQA_CPP_DIALECT 2011
  #elif __cplusplus < 201703L
    #define MUSEQA_CPP_DIALECT 2014
  #elif __cplusplus == 201703L
    #define MUSEQA_CPP_DIALECT 2017
  #elif __cplusplus > 201703L
    #define MUSEQA_CPP_DIALECT 2020
  #endif
#endif

/*
 * Checks the current compiler's C++ language level. As the majority of this software's
 * codebase is written in C++, we must check whether its available or not.
 */
#if !defined(MUSEQA_IGNORE_CPP_DIALECT)
  #if !defined(MUSEQA_CPP_DIALECT) || MUSEQA_CPP_DIALECT < 2017
    #warning Museqa requires at least a C++17 enabled compiler.
  #endif
#endif

/*
 * Checks whether the current compiler is compatible with the recommended prerequisites.
 * Should it not be compatible, then we emit a warning but try compiling anyway.
 */
#if !defined(MUSEQA_IGNORE_COMPILER_CHECK) && MUSEQA_BUILD == MUSEQA_BUILD_PRODUCTION
  #if MUSEQA_HOST_COMPILER == MUSEQA_OPT_HOST_COMPILER_UNKNOWN
    #warning Museqa has not been tested with the current compiler.
  #endif
#endif

/*
 * Defines whether the compilation context is related to the host or device runtime.
 * Code availability or functionality might vary depending on the runtime it is
 * expected to run on.
 */
#if defined(__NVCOMPILER_CUDA__)
  #define MUSEQA_RUNTIME_DEVICE __builtin_is_device_code()
  #define MUSEQA_RUNTIME_HOST (!__builtin_is_device_code())
  #define MUSEQA_INCLUDE_DEVICE_CODE 1
  #define MUSEQA_INCLUDE_HOST_CODE 1
#elif defined(__CUDA_ARCH__)
  #define MUSEQA_RUNTIME_DEVICE 1
  #define MUSEQA_RUNTIME_HOST 0
  #define MUSEQA_INCLUDE_DEVICE_CODE 1
  #define MUSEQA_INCLUDE_HOST_CODE (MUSEQA_HOST_COMPILER == MUSEQA_OPT_HOST_COMPILER_CLANG)
#else
  #define MUSEQA_RUNTIME_DEVICE 0
  #define MUSEQA_RUNTIME_HOST 1
  #define MUSEQA_INCLUDE_DEVICE_CODE 0
  #define MUSEQA_INCLUDE_HOST_CODE 1
#endif

/*
 * Determines whether the software should run in unsafe mode. By default, safe mode
 * is turned off in performance builds to extract maximum possible speedup.
 */
#if MUSEQA_BUILD == MUSEQA_BUILD_PERFORMANCE || MUSEQA_RUNTIME_DEVICE == 1
  #if !defined(MUSEQA_MODE_UNSAFE)
    #define MUSEQA_MODE_UNSAFE 1
  #endif
#endif

/*
 * Since `__host__` and `__device__` annotations are only relevant for CUDA code,
 * we define them to empty strings when CUDA is not being compiled. This allows
 * the use of these annotations without needing to care whether they'll be known
 * by the compiler or not.
 */
#if !defined(__CUDACC__)
  #define __host__
  #define __device__
  #define __forceinline__
#endif

/*
 * Macros for applying annotations and qualifiers to functions and methods. As the
 * minimum required language version is C++17, we assume it is guaranteed that all
 * compilers will have `inline` and `constexpr` implemented.
 */
#define MUSEQA_CUDA_ENABLED __host__ __device__
#define MUSEQA_INLINE MUSEQA_CUDA_ENABLED inline
#define MUSEQA_CONSTEXPR MUSEQA_INLINE constexpr

/**
 * Defines the namespace in which the library lives. This might be overriden if
 * the default namespace value is already in use.
 * @since 1.0
 */
#if defined(MUSEQA_OVERRIDE_NAMESPACE)
  #define MUSEQA_NAMESPACE MUSEQA_OVERRIDE_NAMESPACE
#else
  #define MUSEQA_NAMESPACE museqa
#endif

/**
 * This macro is used to open the `museqa::` namespace block and must not be in
 * any way overriden. This namespace must not be prefixed by any other namespaces
 * to avoid problems when allowing the use some of the library's facilities to with
 * STL's algorithms, structures or constructions.
 * @since 1.0
 */
#define MUSEQA_BEGIN_NAMESPACE   \
    namespace MUSEQA_NAMESPACE { \
        inline namespace v1 {    \
            namespace museqa = MUSEQA_NAMESPACE;

/**
 * This macro is used to close the `museqa::` namespace block and must not be in
 * any way overriden.
 * @since 1.0
 */
#define MUSEQA_END_NAMESPACE     \
    }}
