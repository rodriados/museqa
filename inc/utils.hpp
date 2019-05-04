/** 
 * Multiple Sequence Alignment utilities header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef UTILS_HPP_INCLUDED
#define UTILS_HPP_INCLUDED

#include <cstddef>
#include <utility>

/*
 * Creation of conditional macros that allow CUDA declarations to be used
 * seamlessly throughout the code without any problems.
 */
#if !defined(__host__) && !defined(__device__)
  #define __host__
  #define __device__
#endif

#if defined(__CUDACC__) && !defined(msa_compile_cuda)
  #define msa_compile_cuda 1
#endif

/**#@+
 * Forms the logical conjunction of the given type traits, effectively performing
 * a logical AND in the sequence of traits.
 * @since 0.1.1
 */
template <typename ...T>
struct AllOf : std::true_type
{};

template <typename T>
struct AllOf<T> : T
{};

template <typename T, typename ...U>
struct AllOf<T, U...> : std::conditional<bool(T::value), AllOf<U...>, T>::type
{};
/**#@-*/

/**#@+
 * Forms the logical disjunction of the given type traits, effectively performing
 * a logical OR in the sequence of traits.
 * @since 0.1.1
 */
template <typename ...T>
struct OneOf : std::false_type
{};

template <typename T>
struct OneOf<T> : T
{};

template <typename T, typename ...U>
struct OneOf<T, U...> : std::conditional<bool(T::value), T, OneOf<U...>>::type
{};
/**#@-*/

/**
 * A memory aligned storage container.
 * @tparam S The number of bytes in storage.
 * @tparam A The byte alignment the storage should use.
 * @since 0.1.1
 */
template <size_t S, size_t A>
struct AlignedStorage
{
    alignas(A) char storage[S]; /// The aligned storage container.
};

/**#@+
 * Represents and generates a type index sequence.
 * @tparam I The index sequence.
 * @tparam N The sequence size.
 * @since 0.1.1
 */
template <size_t ...I>
struct Indexer
{
    using type = Indexer;
};

template <>
struct Indexer<0>
{
    using type = Indexer<>;
};

template <>
struct Indexer<1>
{
    using type = Indexer<0>;
};

template <size_t N>
struct Indexer<N>
{
    /**
     * Concatenates two type index sequences into one.
     * @tparam I The first index sequence to merge.
     * @tparam J The second index sequence to merge.
     * @return The concatenated index sequence.
     */
    template <size_t ...I, size_t ...J>
    static constexpr auto concat(Indexer<I...>, Indexer<J...>) noexcept
    -> typename Indexer<I..., sizeof...(I) + J...>::type;

    using type = decltype(concat(
            typename Indexer<N / 2>::type {}
        ,   typename Indexer<N - N / 2>::type {}
        ));
};
/**#@-*/

/**
 * The type index sequence generator of given size.
 * @tparam N The index sequence size.
 * @since 0.1.1
 */
template <size_t N>
using IndexerG = typename Indexer<N>::type;

/**
 * Purifies the type to its base, removing all extents it might have.
 * @tparam T The type to have its base extracted.
 * @since 0.1.1
 */
template <typename T>
using Base = typename std::remove_extent<T>::type;

/**
 * Purifies an array type to its base.
 * @tparam T The type to be purified.
 * @since 0.1.1
 */
template <typename T>
using Pure = typename std::conditional<
        std::is_array<T>::value && !std::extent<T>::value
    ,   Base<T>
    ,   T
    >::type;

/**
 * Represents a function pointer type.
 * @tparam F The function signature type.
 * @since 0.1.1
 */
template <typename F>
using Functor = typename std::enable_if<std::is_function<F>::value, F*>::type;

/**
 * Returns the first type unchanged. This is useful to produce a repeating list
 * of the given type.
 * @tpatam T The type to return.
 * @tparam N Unused.
 * @since 0.1.1
 */
template <typename T, size_t N>
using Identity = T;

#endif