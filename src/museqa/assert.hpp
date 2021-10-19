/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A scoped assertion function implementation, sensible to safety mode.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <utility>

#include <museqa/environment.h>
#include <museqa/utility.hpp>
#include <museqa/exception.hpp>

MUSEQA_DISABLE_GCC_WARNING_BEGIN("-Wattributes")

/*
 * Creates an annotation for an signalling a cold-path to be taken by an if-statement.
 * As we're asserting whether a pre-condition is met or not, we should always favor
 * the likelihood that the condition is indeed met. But if something unexpected
 * unfortunately happens, we can pay the higher cost for messing with the processor's
 * branch predictions in exchange to a slight performance improvement when everything
 * goes as expected, the great majority of times.
 */
#if defined(__has_cpp_attribute) && __has_cpp_attribute(unlikely)
  #define __museqa_unlikely__(condition)                                        \
    ((condition)) [[unlikely]]
#elif MUSEQA_HOST_COMPILER == MUSEQA_HOST_COMPILER_GCC
  #define __museqa_unlikely__(condition)                                        \
    (__builtin_expect((condition), 0))
#else
  #define __museqa_unlikely__(condition)                                        \
    ((condition))
#endif

MUSEQA_BEGIN_NAMESPACE

/**
 * Asserts whether given a condition is met, and throws an exception otherwise.
 * This function acts just like an assertion, but throwing our own exception.
 * @note The name `assert` is not used because it is reserved by some compilers.
 * @tparam E The exception type to be raised in case of error.
 * @tparam T The exception's parameters' types.
 * @param fact The condition that is expected to be a fact, to be true.
 * @param params The assertion exception's parameters.
 */
template <typename E = museqa::exception, typename ...T>
__host__ __device__ inline constexpr void ensure(bool fact, T&&... params) __museqasafe__
{
    static_assert(std::is_base_of<museqa::exception, E>::value, "only exception types are throwable");

  #if !defined(MUSEQA_MODE_UNSAFE)
    if __museqa_unlikely__ (!fact) {
        throw E (std::forward<decltype(params)>(params)...);
    }
  #endif
}

MUSEQA_END_NAMESPACE

#undef __museqa_unlikely__
MUSEQA_DISABLE_GCC_WARNING_END("-Wattributes")
