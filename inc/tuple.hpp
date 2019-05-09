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
    __host__ __device__ inline constexpr TupleLeaf(T& value) noexcept
    :   value {value}
    {}

    /**
     * Constructs a new tuple leaf.
     * @tparam U A convertible type for leaf.
     * @param value The value to be held by the leaf.
     */
    template <typename U>
    __host__ __device__ inline constexpr TupleLeaf(const U& value) noexcept
    :   value {static_cast<T>(value)}
    {}

    /**
     * Constructs a new tuple leaf.
     * @tparam U A convertible type for leaf.
     * @param value The value to be moved to the leaf.
     */
    template <typename U>
    __host__ __device__ inline constexpr TupleLeaf(U&& value) noexcept
    :   value {static_cast<T&&>(std::move(value))}
    {}

    /**
     * Constructs a new tuple leaf by copying a foreign tuple.
     * @tparam U A convertible foreign type for leaf.
     * @param leaf The leaf to copy contents from.
     */
    template <typename U>
    __host__ __device__ inline constexpr TupleLeaf(const TupleLeaf<I, U>& leaf) noexcept
    :   value {static_cast<T>(leaf.value)}
    {}

    /**
     * Constructs a new tuple leaf by moving a foreign tuple.
     * @tparam U A convertible foreign type for leaf.
     * @param leaf The leaf to move contents from.
     */
    template <typename U>
    __host__ __device__ inline constexpr TupleLeaf(TupleLeaf<I, U>&& leaf) noexcept
    :   value {static_cast<T&&>(std::move(leaf.value))}
    {}

    __host__ __device__ inline TupleLeaf& operator=(const TupleLeaf&) = default;
    __host__ __device__ inline TupleLeaf& operator=(TupleLeaf&&) = default;
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
struct BaseTuple<Indexer<I...>, T...> : public TupleLeaf<I, T>...
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
    :   TupleLeaf<I, T> {value}...
    {}

    /**
     * This constructor sets every base member with its corresponding value.
     * @tparam U A list of convertible types for every base member.
     * @param value The list of values for members.
     */
    template <typename ...U, typename = typename std::enable_if<
            utils::all<std::is_convertible<U, T>...>()
        >::type >
    __host__ __device__ inline BaseTuple(const U&... value) noexcept
    :   TupleLeaf<I, T> {static_cast<T>(value)}...
    {}

    /**
     * This constructor sets every base member with its corresponding value.
     * @tparam U A list of convertible types for every base member.
     * @param value The list of values for members.
     */
    template <typename ...U, typename = typename std::enable_if<
            utils::all<std::is_convertible<U, T>...>()
        >::type >
    __host__ __device__ inline BaseTuple(U&&... value) noexcept
    :   TupleLeaf<I, T> {static_cast<T&&>(std::move(value))}...
    {}

    __host__ __device__ inline BaseTuple& operator=(const BaseTuple&) = default;
    __host__ __device__ inline BaseTuple& operator=(BaseTuple&&) = default;
};

template <>
struct BaseTuple<Indexer<>>
{
    static constexpr size_t count = 0;      /// The size of the tuple.
};
/**#@-*/

namespace tuple
{
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

    /**
     * Modifies the value held by a tuple leaf.
     * @tparam I The requested leaf index.
     * @tparam T The type of requested leaf member.
     * @tparam U The type of new leaf value.
     * @param leaf The selected tuple leaf member.
     * @param value The value to copy to leaf.
     */
    template <size_t I, typename T, typename U>
    __host__ __device__ inline void set(TupleLeaf<I, T>& leaf, const U& value) noexcept
    {
        leaf.value = value;
    }

    /**
     * Modifies the value held by a tuple leaf.
     * @tparam I The requested leaf index.
     * @tparam T The type of requested leaf member.
     * @tparam U The type of new leaf value.
     * @param leaf The selected tuple leaf member.
     * @param value The value to move to leaf.
     */
    template <size_t I, typename T, typename U>
    __host__ __device__ inline void set(TupleLeaf<I, T>& leaf, U&& value) noexcept
    {
        leaf.value = std::move(value);
    }
};

/**
 * A tuple is responsible for holding a list of elements of possible different
 * types with a known number of elements.
 * @tparam T The tuple's list of member types.
 * @since 0.1.1
 */
template <typename ...T>
class Tuple : public BaseTuple<IndexerG<sizeof...(T)>, T...>
{
    public:
        __host__ __device__ inline constexpr Tuple() noexcept = default;
        __host__ __device__ inline constexpr Tuple(const Tuple&) noexcept = default;
        __host__ __device__ inline constexpr Tuple(Tuple&&) noexcept = default;

        using BaseTuple<IndexerG<sizeof...(T)>, T...>::BaseTuple;

        /**
         * Creates a new tuple from a tuple of different base types.
         * @tparam U The types of tuple instance to copy from.
         * @param other The tuple the values must be copied from.
         */
        template <typename ...U>
        __host__ __device__ inline Tuple(const Tuple<U...>& other) noexcept
        {
            operator=(other);
        }

        /**
         * Creates a new tuple from moving a tuple of different base types.
         * @tparam U The types of tuple instance to copy from.
         * @param other The tuple the values must be copied from.
         */
        template <typename ...U>
        __host__ __device__ inline Tuple(Tuple<U...>&& other) noexcept
        {
            operator=(std::move(other));
        }

        /**
         * Copies values from another tuple instance.
         * @param other The tuple the values must be copied from.
         * @return This object instance.
         */
        __host__ __device__ inline Tuple& operator=(const Tuple& other)
        {
            return copy(IndexerG<sizeof...(T)> {}, other);
        }

        /**
         * Copies the values from a foreign tuple instance.
         * @tparam U The types of tuple instance to copy from.
         * @param other The tuple the values must be copied from.
         * @return This object instance.
         */
        template <typename ...U>
        __host__ __device__ inline Tuple& operator=(const Tuple<U...>& other)
        {
            return copy(IndexerG<sizeof...(T)> {}, other);
        }

        /**
         * Moves the values from another tuple instance.
         * @param other The tuple the values must be moved from.
         * @return This object instance.
         */
        __host__ __device__ inline Tuple& operator=(Tuple&& other)
        {
            return copy(IndexerG<sizeof...(T)> {}, std::move(other));
        }

        /**
         * Moves the values from a foreign tuple instance.
         * @tparam U The types of tuple instance to copy from.
         * @param other The tuple the values must be moved from.
         * @return This object instance.
         */
        template <typename ...U>
        __host__ __device__ inline Tuple& operator=(Tuple<U...>&& other)
        {
            return copy(IndexerG<sizeof...(T)> {}, std::move(other));
        }

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

        /**
         * Sets a member value by its index.
         * @tparam I The index of requested member.
         * @tparam U The new value type.
         */
        template <size_t I, typename U>
        __host__ __device__ inline void set(const U& value) noexcept
        {
            tuple::set<I>(*this, value);
        }

        /**
         * Sets a member value by its index.
         * @tparam I The index of requested member.
         * @tparam U The new value type.
         */
        template <size_t I, typename U>
        __host__ __device__ inline void set(U&& value) noexcept
        {
            tuple::set<I>(*this, std::move(value));
        }

    protected:
        /**
         * Recursion basis for copy operation.
         * @tparam U The foreign tuple type.
         * @return This object instance.
         */
        template <typename U>
        __host__ __device__ inline Tuple& copy(Indexer<>, const U&) noexcept
        {
            return *this;
        }

        /**
         * Copies values from a foreign tuple instance.
         * @tpatam I The first member index to be copied.
         * @tparam J The following member indeces to copy.
         * @tparam U The foreign tuple base types.
         * @return This object instance.
         */
        template <size_t I, size_t ...J, typename ...U>
        __host__ __device__ inline Tuple& copy(Indexer<I, J...>, const Tuple<U...>& other) noexcept
        {
            set<I>(other.get<I>());
            return copy(Indexer<J...> {}, other);
        }

        /**
         * Moves values from a foreign tuple instance.
         * @tpatam I The first member index to be copied.
         * @tparam J The following member indeces to copy.
         * @tparam U The foreign tuple base types.
         * @return This object instance.
         */
        template <size_t I, size_t ...J, typename ...U>
        __host__ __device__ inline Tuple& copy(Indexer<I, J...>, Tuple<U...>&& other) noexcept
        {
            set<I>(std::move(other.get<I>()));
            return copy(Indexer<J...> {}, std::move(other));
        }
};

namespace tuple
{
    /**
     * Creates a tuple with repeated types.
     * @tparam T The type to repeat.
     * @tparam I The number of times to repeat the type.
     */
    template <typename T, size_t ...I>
    constexpr auto repeater(Indexer<I...>) noexcept
    -> Tuple<Identity<T, I>...>;

    template <typename T, size_t N>
    using Repeated = decltype(repeater<T>(IndexerG<N> {}));
};

/**
 * Creates a tuple with repeated types.
 * @tparam T The type to be repeated.
 * @tparam N The number of times the type shall repeat.
 * @since 0.1.1
 */
template <typename T, size_t N>
class TupleN : public tuple::Repeated<T, N>
{
    public:
        using Tuple = tuple::Repeated<T, N>;

        __host__ __device__ inline constexpr TupleN() noexcept = default;
        __host__ __device__ inline constexpr TupleN(const TupleN&) noexcept = default;
        __host__ __device__ inline constexpr TupleN(TupleN&&) noexcept = default;

        using Tuple::Tuple;

        /**
         * Initializes a new tuple from an array.
         * @param arr The array to initialize tuple.
         */
        __host__ __device__ inline TupleN(Pure<T> *arr) noexcept
        :   Tuple {getElements(IndexerG<N>{}, arr)}
        {}

        /**
         * Initializes a new tuple from a const array.
         * @param arr The array to initialize tuple.
         */
        template <typename U = T, typename = typename std::enable_if<
                !std::is_reference<U>::value
            >::type >
        __host__ __device__ inline TupleN(const Pure<T> *arr) noexcept
        :   Tuple {getElements(IndexerG<N>{}, arr)}
        {}

        __host__ __device__ inline TupleN& operator=(const TupleN&) = default;
        __host__ __device__ inline TupleN& operator=(TupleN&&) = default;

        using Tuple::operator=;

    protected:
        /**
         * A helper function to map array values to the underlying tuple.
         * @param arr The array to inline.
         * @return The created tuple.
         */
        template <size_t ...I, typename U>
        __host__ __device__ inline static auto getElements(Indexer<I...>, U *arr) noexcept
        -> Tuple
        {
            return {arr[I]...};
        }
};

/**
 * The type of a tuple element.
 * @tparam T The target tuple.
 * @tparam I The index of tuple element.
 * @since 0.1.1
 */
template <typename T, size_t I>
using TupleElement = decltype(tuple::get<I>(std::declval<T>()));

namespace tuple
{
    /**#@+
     * Gathers variable or array references into a tuple, allowing them to
     * capture values directly from value tuples.
     * @tparam T The gathered variables types.
     * @tparam N When an array, the size must be fixed.
     * @param arg The gathered variables references.
     * @return The new tuple of references.
     */
    template <typename ...T>
    __host__ __device__ inline auto tie(T&... arg) noexcept -> Tuple<T&...>
    {
        return {arg...};
    }

    template <typename T, size_t N>
    __host__ __device__ inline auto tie(T (&arg)[N]) noexcept -> TupleN<T&, N>
    {
        return {arg};
    }
    /**#@-*/
};

#endif