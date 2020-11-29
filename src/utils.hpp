/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Miscellaneous utilities and helper functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "environment.h"

/*
 * Definition of CUDA function flags for host code, so we don't need to care about
 * which compiler is working on the file when using these flags.
 */
#if !defined(__host__) && !defined(__device__)
  #define __host__
  #define __device__
#endif

#include "oeis.hpp"
#include "functor.hpp"
#include "operators.hpp"

namespace museqa
{
    /**
     * A general memory storage container.
     * @tparam S The number of bytes in storage.
     * @tparam A The byte alignment the storage should use.
     * @since 0.1.1
     */
    template <size_t S, size_t A = S>
    struct alignas(A) storage
    {
        alignas(A) char storage[S];     /// The storage container.
    };

    /**
     * A general range container.
     * @tparam T The range's elements type.
     * @since 0.1.1
     */
    template <typename T>
    struct range
    {
        T offset, total;
    };

    /**
     * Purifies the type to its base, removing all extents it might have.
     * @tparam T The type to have its base extracted.
     * @since 0.1.1
     */
    template <typename T>
    using base = typename std::remove_extent<T>::type;

    /**
     * Purifies an array type to its base.
     * @tparam T The type to be purified.
     * @since 0.1.1
     */
    template <typename T>
    using pure = typename std::conditional<
            !std::is_array<T>::value || std::extent<T>::value
        ,   typename std::remove_reference<T>::type
        ,   base<T>
        >::type;

    /**
     * Returns the type unchanged. This is useful to produce a repeating list of
     * the type given first.
     * @tpatam T The type to return.
     * @since 0.1.1
     */
    template <typename T, size_t = 0>
    using identity = T;

    /**#@+
     * Represents and generates a type index sequence.
     * @tparam I The index sequence.
     * @tparam L The length of sequence to generate.
     * @since 0.1.1
     */
    template <size_t ...I>
    struct indexer
    {
        /**
         * The indexer sequence type.
         * @since 0.1.1
         */
        using type = indexer;
    };

    template <>
    struct indexer<0>
    {
        /**
         * The indexer base generator type.
         * @since 0.1.1
         */
        using type = indexer<>;
    };

    template <>
    struct indexer<1>
    {
        /**
         * The indexer base generator type.
         * @since 0.1.1
         */
        using type = indexer<0>;
    };

    template <size_t L>
    struct indexer<L>
    {
        /**
         * Concatenates two type index sequences into one.
         * @tparam I The first index sequence to merge.
         * @tparam J The second index sequence to merge.
         * @return The concatenated index sequence.
         */
        template <size_t ...I, size_t ...J>
        __host__ __device__ static constexpr auto concat(indexer<I...>, indexer<J...>) noexcept
        -> typename indexer<I..., sizeof...(I) + J...>::type;

        /**
         * The indexer generator type.
         * @since 0.1.1
         */
        using type = decltype(concat(
                typename indexer<L / 2>::type {}
            ,   typename indexer<L - L / 2>::type {}
            ));
    };
    /**#@-*/

    /**
     * The type index sequence generator of given size.
     * @tparam L The length of index sequence to generate.
     * @since 0.1.1
     */
    template <size_t L>
    using indexer_g = typename indexer<L>::type;

    namespace utils
    {
        using namespace op;
        using namespace oeis;

        /**
         * Calculates the number of possible pair combinations with given number.
         * @param n The total number of objects to be combined.
         * @return The number of possible pair combinations.
         */
        template <typename T>
        __host__ __device__ inline auto nchoose(const T& n) noexcept
        -> typename std::enable_if<std::is_integral<T>::value, T>::type
        {
            return oeis::a000217(n - 1);
        }

        /**
         * Calculates the range of a partition in a collection of given total size.
         * @tparam T The given collection size's type.
         * @param total The collection's size.
         * @param count The total number of partitions on the collection.
         * @param offset The requested range's partition index.
         * @return The range of the requested collection's partition.
         */
        template <typename T>
        __host__ __device__ inline auto partition(T total, size_t count, size_t offset = 0) noexcept
        -> typename std::enable_if<std::is_integral<T>::value, range<T>>::type
        {
            const auto quo = total / count;
            const auto rem = total % count;
            return {quo * offset + min(rem, offset), quo + (rem > offset)};
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
            )
        {
            auto aux = std::move(a);
            a = std::move(b);
            b = std::move(aux);
        }

        /**
         * Swaps the elements of two arrays of same type and size.
         * @tparam T The arrays' elements' type.
         * @tparam N The arrays' size.
         * @param a The first array to have its elements swapped.
         * @param b The second array to have its elements swapped.
         */
        template <typename T, size_t N>
        __host__ __device__ inline void swap(T (&a)[N], T (&b)[N])
            noexcept(noexcept(swap(*a, *b)))
        {
            for(size_t i = 0; i < N; ++i)
                swap(a[i], b[i]);
        }

        /**
         * Converts the given string value to any other generic type.
         * @tparam T The target type to convert to.
         * @param value The value to be converted.
         * @return The converted value to the requested type.
         */
        template <typename T>
        inline auto convert(const std::string& value)
        -> typename std::enable_if<std::is_convertible<std::string, T>::value, T>::value
        {
            return T (value);
        }

        /**
         * Converts the given string value to an integral type.
         * @tparam T The target type to convert to.
         * @param value The value to be converted.
         * @return The converted value to the requested type.
         * @throw std::exception Error detected during operation.
         */
        template <typename T>
        inline auto convert(const std::string& value)
        -> typename std::enable_if<std::is_integral<T>::value, T>::type
        {
            return static_cast<T>(std::stoull(value));
        }

        /**
         * Converts the given string value to a floating point type.
         * @tparam T The target type to convert to.
         * @param value The value to be converted.
         * @return The converted value to the requested type.
         * @throw std::exception Error detected during operation.
         */
        template <typename T>
        inline auto convert(const std::string& value)
        -> typename std::enable_if<std::is_floating_point<T>::value, T>::type
        {
            return static_cast<T>(std::stold(value));
        }

        /**
         * Retrieves the given file's name's extension.
         * @param filename The file to have its extension retrieved.
         * @return The given file's extension.
         */
        inline auto extension(const std::string& filename) noexcept -> std::string
        {
            if(filename.size()) return filename.substr(filename.find_last_of('.') + 1);
            return std::string {};
        }
    }
}
