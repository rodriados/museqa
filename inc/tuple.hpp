/** 
 * Multiple Sequence Alignment tuple header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef TUPLE_HPP_INCLUDED
#define TUPLE_HPP_INCLUDED

#include <cstdint>
#include <utility>

#include "utils.hpp"

namespace tuple
{
    /**
     * Represents a tuple leaf, which holds a value.
     * @tparam I The index of the tuple's leaf.
     * @tparam T The type of the tuple's leaf member.
     * @since 0.1.1
     */
    template <size_t I, typename T>
    struct TupleLeaf
    {
        T value;        /// The value held by this tuple leaf.

        __host__ __device__ inline constexpr TupleLeaf() noexcept = default;
        __host__ __device__ inline constexpr TupleLeaf(const TupleLeaf&) noexcept = default;
        __host__ __device__ inline constexpr TupleLeaf(TupleLeaf&&) noexcept = default;

        /**
         * Constructs a new tuple leaf.
         * @param value The value to be held by the leaf.
         */
        __host__ __device__ inline constexpr TupleLeaf(const T& value) noexcept
        :   value {value}
        {}

        __host__ __device__ inline TupleLeaf& operator=(const TupleLeaf&) noexcept = default;
        __host__ __device__ inline TupleLeaf& operator=(TupleLeaf&&) noexcept = default;
    };

    /**
     * Retrieves the requested tuple leaf and returns its value.
     * @param leaf The selected tuple leaf member.
     * @tparam I The requested leaf index.
     * @tparam T The type of requested leaf member.
     * @return The leaf's value.
     */
    template <size_t I, typename T>
    __host__ __device__ inline constexpr const T& get(const TupleLeaf<I, T>& leaf) noexcept
    {
        return leaf.value;
    }
};

/**#@+
 * The base for a type tuple.
 * @tparam I The indeces for the tuple members.
 * @tparam T The types of the tuple members.
 * @since 0.1.1
 */
template <typename I, typename ...T>
struct BaseTuple;

template <size_t ...I, typename ...T>
struct BaseTuple<Index<I...>, T...> : public tuple::TupleLeaf<I, T>...
{
    static constexpr size_t count = sizeof...(I);   /// The size of the tuple.

    __host__ __device__ inline constexpr BaseTuple() noexcept = default;
    __host__ __device__ inline constexpr BaseTuple(const BaseTuple&) noexcept = default;
    __host__ __device__ inline constexpr BaseTuple(BaseTuple&&) noexcept = default;

    /**
     * This constructor sets every base member with its corresponding value.
     * @param value The list of values for members.
     */
    __host__ __device__ inline constexpr BaseTuple(const T&... value) noexcept
    :   tuple::TupleLeaf<I, T> {value}...
    {}

    __host__ __device__ inline BaseTuple& operator=(const BaseTuple&) noexcept = default;
    __host__ __device__ inline BaseTuple& operator=(BaseTuple&&) noexcept = default;
};

template <>
struct BaseTuple<Index<>>
{
    static constexpr size_t count = 0;      /// The size of the tuple.
};
/**#@-*/

/**
 * A tuple is responsible for holding a list of elements of possible different
 * types with a known number of elements.
 * @tparam T The tuple's list of member types.
 * @since 0.1.1
 */
template <typename ...T>
struct Tuple : public BaseTuple<typename IndexGen<sizeof...(T)>::type, T...>
{
    __host__ __device__ inline constexpr Tuple() noexcept = default;
    __host__ __device__ inline constexpr Tuple(const Tuple&) noexcept = default;
    __host__ __device__ inline constexpr Tuple(Tuple&&) noexcept = default;

    using BaseTuple<typename IndexGen<sizeof...(T)>::type, T...>::BaseTuple;

    __host__ __device__ inline Tuple& operator=(const Tuple&) noexcept = default;
    __host__ __device__ inline Tuple& operator=(Tuple&&) noexcept = default;

    /**
     * Gets value from member by index.
     * @tparam I The index of requested member.
     * @return The member's value.
     */
    template <size_t I>
    __host__ __device__ inline constexpr auto get() const noexcept
    -> decltype(tuple::get<I>(std::declval<Tuple>()))
    {
        return tuple::get<I>(*this);
    }
};

/**
 * The type of a tuple element.
 * @tparam I The index of tuple element.
 * @tparam T The target tuple.
 * @since 0.1.1
 */
template <size_t I, typename T>
using TupleElement = typename std::remove_reference<
        decltype(tuple::get<I>(std::declval<T>()))
    >::type;

#endif