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

/**
 * A memory aligned storage container.
 * @tparam S The number of elements in storage.
 * @tparam A The alignment the storage should use.
 * @since 0.1.1
 */
template <size_t S, size_t A>
struct AlignedStorage
{
    alignas(A) char storage[S]; /// The aligned storage container.
};

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

/**
 * Represents an index sequence in its type.
 * @tparam I The index sequence.
 * @since 0.1.1
 */
template <size_t ...I>
struct Index
{
    using type = Index;
    static constexpr size_t size = sizeof...(I);
};

namespace utils
{
    /**
     * Concatenates two index sequences into one.
     * @tparam I The first sequence to be concatenated.
     * @tparam J The second sequence to be concatenated.
     * @return The type of concatenated sequences.
     */
    template <size_t ...I, size_t ...J>
    __host__ __device__ static constexpr auto concat(Index<I...>, Index<J...>) noexcept
    -> Index<I..., sizeof...(I) + J...>;
};

/**#@+
 * Generates a tuple type index sequence.
 * @tparam N The size of sequence to generate.
 * @since 0.1.1
 */
template <size_t N>
struct IndexGen : public decltype(utils::concat(
        typename IndexGen<N / 2>::type ()
    ,   typename IndexGen<N - N / 2>::type ()
    ))
{};

template <>
struct IndexGen<0> : public Index<>
{};

template <>
struct IndexGen<1> : public Index<0>
{};
/**#@-*/


#endif