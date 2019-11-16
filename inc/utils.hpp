/**
 * Multiple Sequence Alignment utilities header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef UTILS_HPP_INCLUDED
#define UTILS_HPP_INCLUDED

#include <cstddef>
#include <cstdint>
#include <utility>

/*
 * Creation of conditional macros that allow CUDA declarations to be used seamlessly
 * throughout the code without any problems.
 */
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
  #define onlydevice 1
#else
  #define onlyhost 1
#endif

/*
 * Creation of conditional macro for Cython code, allowing the creation of specialized
 * code or the removal of unnecessary functions.
 */
#if defined(msa_compile_cython)
  #define onlycython
#endif

/*
 * Definition of CUDA function flags for host code, so we don't need to care about
 * which compiler is working on the file when using these flags.
 */
#if !defined(__host__) && !defined(__device__)
  #define __host__
  #define __device__
#endif

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

/**#@+
 * Wraps a function pointer into a functor.
 * @tparam F The full function signature type.
 * @tparam R The function return type.
 * @tparam P The function parameter types.
 * @since 0.1.1
 */
template <typename F>
class functor
{
    static_assert(std::is_function<F>::value, "a functor must have a function signature type");
};

template <typename R, typename ...P>
class functor<R(P...)>
{
    public:
        using return_type = R;                  /// The functor's return type.
        using function_type = R (*)(P...);      /// The functor's raw pointer type.

    protected:
        function_type mfunction = nullptr;      /// The raw functor's pointer.

    public:
        __host__ __device__ inline constexpr functor() noexcept = default;
        __host__ __device__ inline constexpr functor(const functor&) noexcept = default;
        __host__ __device__ inline constexpr functor(functor&&) noexcept = default;

        /**
         * Instantiates a new functor.
         * @param function The function pointer to be encapsulated by functor.
         */
        __host__ __device__ inline constexpr functor(function_type function) noexcept
        :   mfunction {function}
        {}

        __host__ __device__ inline functor& operator=(const functor&) noexcept = default;
        __host__ __device__ inline functor& operator=(functor&&) noexcept = default;

        /**
         * The functor call operator.
         * @tparam T The given parameter types.
         * @param param The given functor parameters.
         * @return The functor return value.
         */
        template <typename ...T>
        __host__ __device__ inline constexpr return_type operator()(T&&... param) const
        {
            return (mfunction)(param...);
        }

        /**
         * Allows the raw functor type to be directly accessed or called.
         * @return The raw function pointer.
         */
        __host__ __device__ inline constexpr function_type operator&() const
        {
            return mfunction;
        }

        /**
         * Checks whether the functor is empty or not.
         * @return Is the functor empty?
         */
        __host__ __device__ inline constexpr bool empty() const noexcept
        {
            return (mfunction == nullptr);
        }
};
/**#@-*/

#include <operator.hpp>

namespace utils
{
    /**#@+
     * Checks whether all given values are true.
     * @param head The first value to test.
     * @param tail The following values to test.
     * @return Are all values true?
     */
    __host__ __device__ inline constexpr auto all() noexcept -> bool
    {
        return true;
    }

    template <typename T, typename ...U>
    __host__ __device__ inline constexpr auto all(T&& head, U&&... tail) noexcept -> bool
    {
        return static_cast<bool>(head) ? all(tail...) : false;
    }
    /**#@-*/

    /**#@+
     * Checks whether at least one given value is true.
     * @param head The first value to test.
     * @param tail The following values to test.
     * @return Is at least one value true?
     */
    __host__ __device__ inline constexpr auto any() noexcept -> bool
    {
        return false;
    }

    template <typename T, typename ...U>
    __host__ __device__ inline constexpr auto any(T&& head, U&&... tail) noexcept -> bool
    {
        return static_cast<bool>(head) ? true : any(tail...);
    }
    /**#@-*/

    /**
     * Checks whether none of given values is true.
     * @param value All the values to be tested.
     * @return Are all values false?
     */
    template <typename ...U>
    __host__ __device__ inline constexpr auto none(U&&... tail) noexcept -> bool
    {
        return !any(tail...);
    }

    /**
     * Calculates the number of possible pair combinations with given number.
     * @param count The number of objects to be combinated.
     * @return The number of possible pair combinations.
     */
    template <typename T>
    __host__ __device__ inline constexpr auto nchoose(const T& count) noexcept
    -> typename std::enable_if<std::is_integral<T>::value, T>::type
    {
        return (count * (count - 1)) >> 1;
    }

    /**#@+
     * Swaps the contents of two variables of same type
     * @param a The first variable to have its contents swapped.
     * @param b The second variable to have its contents swapped.
     */
    template <typename T>
    __host__ __device__ inline void swap(T& a, T& b)
        noexcept(std::is_nothrow_move_constructible<T>::value &&  std::is_nothrow_move_assignable<T>::value)
    {
        T aux = std::move(a);
        a = std::move(b);
        b = std::move(aux);
    }

    template <typename T, size_t N>
    __host__ __device__ inline void swap(T (&a)[N], T (&b)[N])
        noexcept(noexcept(swap(*a, *b)))
    {
        for(size_t i = 0; i < N; ++i)
            swap(a[i], b[i]);
    }
    /**#@-*/
}

#endif