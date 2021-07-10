/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Miscellaneous utilities and helper functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <cstdint>
#include <utility>

#include <museqa/environment.h>

/*
 * Definition of CUDA function flags for host code, so we don't need to care about
 * which compiler is working on the file when using these flags.
 */
#if !defined(__host__) && !defined(__device__)
  #define __host__
  #define __device__
#endif

#include <museqa/utility/operators.hpp>

namespace museqa
{
    /**
     * Informs whether unsafe mode is turned on.
     * @since 1.0
     */
    enum : bool
    {
      #if !defined(MUSEQA_UNSAFE)
        unsafe = false
      #else
        unsafe = true
      #endif
    };

    /**
     * A general memory storage container.
     * @tparam S The number of bytes in storage.
     * @tparam A The byte alignment the storage should use.
     * @since 1.0
     */
    template <size_t S, size_t A = S>
    struct alignas(A) storage
    {
        alignas(A) char storage[S];
    };

    /**
     * A general range container.
     * @tparam T The range's elements type.
     * @since 1.0
     */
    template <typename T = int>
    struct range
    {
        T offset, total;
    };

    /**
     * Purifies the type to its base, removing all extents it might have.
     * @tparam T The type to be purified.
     * @since 1.0
     */
    template <typename T>
    using pure = typename std::conditional<
            !std::is_array<T>::value || std::extent<T>::value
          , typename std::remove_reference<T>::type
          , typename std::remove_extent<T>::type
        >::type;

    /**
     * Returns the type unchanged. This is useful to produce a repeating list of the
     * given type parameter.
     * @tpatam T The identity type.
     * @since 1.0
     */
    template <typename T, size_t = 0>
    using identity = T;

    namespace utility
    {
        /**
         * Parses the given string value into any other generic type.
         * @tparam T The target type to parse the string to.
         * @param value The string to be parsed.
         * @return The parsed value to the requested type.
         */
        template <typename T>
        inline auto parse(const std::string& value)
        -> typename std::enable_if<std::is_convertible<std::string, T>::value, T>::type
        {
            return T (value);
        }

        /**
         * Parses the given string value into an integer type.
         * @tparam T The target type to parse the string to.
         * @param value The string to be parsed.
         * @return The parsed value to the requested type.
         * @throw std::exception Error detected during operation.
         */
        template <typename T>
        inline auto parse(const std::string& value)
        -> typename std::enable_if<std::is_integral<T>::value, T>::type
        {
            return static_cast<T>(std::stoull(value));
        }

        /**
         * Parses the given string value into a floating-point type.
         * @tparam T The target type to parse the string to.
         * @param value The string to be parsed.
         * @return The parsed value to the requested type.
         * @throw std::exception Error detected during operation.
         */
        template <typename T>
        inline auto parse(const std::string& value)
        -> typename std::enable_if<std::is_floating_point<T>::value, T>::type
        {
            return static_cast<T>(std::stold(value));
        }

        /**
         * Swaps the contents of two variables of same type
         * @tparam T The variables' type.
         * @param a The first variable to have its contents swapped.
         * @param b The second variable to have its contents swapped.
         */
        template <typename T>
        __host__ __device__ inline void swap(T& a, T& b) noexcept(
            std::is_nothrow_move_constructible<T>::value &&
            std::is_nothrow_move_assignable<T>::value
        ) {
            T x = std::move(a);
              a = std::move(b);
              b = std::move(x);
        }

        /**
         * Swaps the elements of two arrays of same type and size.
         * @tparam T The arrays' elements' swappable type.
         * @tparam N The arrays' size.
         * @param a The first array to have its elements swapped.
         * @param b The second array to have its elements swapped.
         */
        template <typename T, size_t N>
        __host__ __device__ inline void swap(T (&a)[N], T (&b)[N]) noexcept(noexcept(swap(*a, *b)))
        {
            for(size_t i = 0; i < N; ++i)
                swap(a[i], b[i]);
        }

        /**
         * Swallows a variadic list of parameters and returns the first one. This
         * functions is useful when dealing with type-packs.
         * @tparam T The type of the value to be returned.
         * @tparam U The type of the values to ignore.
         * @return The given return value.
         */
        template <typename T, typename ...U>
        __host__ __device__ inline constexpr T swallow(T&& target, U&&...) noexcept
        {
            return target;
        }
    }
}
