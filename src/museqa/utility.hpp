/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Miscellaneous utilities and helper functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>
#include <utility>

#include <museqa/environment.h>

#include <museqa/utility/common.hpp>
#include <museqa/utility/operators.hpp>

MUSEQA_BEGIN_NAMESPACE

/**
 * Informs whether safe mode is turned on. When safe mode is turned on, bound
 * checks and API call validations will be performed.
 * @since 1.0
 */
enum : bool {
  #if !defined(MUSEQA_MODE_UNSAFE)
    safe = true
  #else
    safe = false
  #endif
};

/**
 * Annotation for functions and methods that may throw exceptions when running on
 * safe mode, but that will not perform checks when safe mode is disabled.
 * @since 1.0
 */
#define __museqasafe__ noexcept(!museqa::safe)

/**
 * Annotation for functions and methods that may throw exceptions when running on
 * host code, but that are exception-free when running on device.
 * @since 1.0
 */
#define __devicesafe__ noexcept(MUSEQA_RUNTIME_DEVICE)

/**
 * Returns the type unchanged. This is useful to produce a repeating list of the
 * given type parameter.
 * @tpatam T The identity type.
 * @since 1.0
 */
template <typename T, size_t = 0>
struct identity_t
{
    using type = T;
};

/**
 * A general memory storage container.
 * @tparam S The number of bytes in storage.
 * @tparam A The byte alignment the storage should use.
 * @since 1.0
 */
template <size_t S, size_t A = 8>
struct alignas(A) storage_t
{
    alignas(A) uint8_t storage[S];
};

/**
 * A general range container.
 * @tparam T The range's elements type.
 * @since 1.0
 */
template <typename T = int>
struct range_t
{
    T offset, total;
};

/**
 * Purifies the type to its base, removing all extents it might have.
 * @tparam T The type to be purified.
 * @since 1.0
 */
template <typename T>
using pure_t = typename std::conditional<
        !std::is_array<T>::value || std::extent<T>::value
      , typename std::remove_reference<T>::type
      , typename std::remove_extent<T>::type
    >::type;

/**
 * A type to represent an empty return type. This is essentialy a void-like type
 * that can be instantiated and returned by a function.
 * @since 1.0
 */
struct nothing_t : identity_t<void> {};

MUSEQA_END_NAMESPACE
