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
        :   value {static_cast<T>(std::move(value))}
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
        :   value {static_cast<T>(std::move(leaf.value))}
        {}

        __host__ __device__ inline TupleLeaf& operator=(const TupleLeaf&) = default;
        __host__ __device__ inline TupleLeaf& operator=(TupleLeaf&&) = default;
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
        leaf.value = std::forward<U>(value);
    }

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
                utils::all(std::is_convertible<U, T>{}...)
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
                utils::all(std::is_convertible<U, T>{}...)
            >::type >
        __host__ __device__ inline BaseTuple(U&&... value) noexcept
        :   TupleLeaf<I, T> {std::move(value)}...
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
};

/**
 * A tuple is responsible for holding a list of elements of possible different
 * types with a known number of elements.
 * @tparam T The tuple's list of member types.
 * @since 0.1.1
 */
template <typename ...T>
class Tuple : public tuple::BaseTuple<IndexerG<sizeof...(T)>, T...>
{
    public:
        __host__ __device__ inline constexpr Tuple() noexcept = default;
        __host__ __device__ inline constexpr Tuple(const Tuple&) noexcept = default;
        __host__ __device__ inline constexpr Tuple(Tuple&&) noexcept = default;

        using tuple::BaseTuple<IndexerG<sizeof...(T)>, T...>::BaseTuple;

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
            set<I>(other.template get<I>());
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
            set<I>(std::move(other.template get<I>()));
            return copy(Indexer<J...> {}, std::move(other));
        }
};

namespace tuple
{
    namespace detail
    {
        /**
         * Creates a tuple with repeated types.
         * @tparam T The type to repeat.
         * @tparam I The number of times to repeat the type.
         */
        template <typename T, size_t ...I>
        constexpr auto repeater(Indexer<I...>) noexcept
        -> Tuple<Identity<T, I>...>;
    };
};

/**
 * Creates a tuple with repeated types.
 * @tparam T The type to be repeated.
 * @tparam N The number of times the type shall repeat.
 * @since 0.1.1
 */
template <typename T, size_t N>
class TupleN : public decltype(tuple::detail::repeater<T>(IndexerG<N>()))
{
    public:
        using Tuple = decltype(tuple::detail::repeater<T>(IndexerG<N>()));

        __host__ __device__ inline constexpr TupleN() noexcept = default;
        __host__ __device__ inline constexpr TupleN(const TupleN&) noexcept = default;
        __host__ __device__ inline constexpr TupleN(TupleN&&) noexcept = default;

        using Tuple::Tuple;

        /**
         * Initializes a new tuple from an array.
         * @param arr The array to initialize tuple.
         */
        __host__ __device__ inline constexpr TupleN(Pure<T> *arr) noexcept
        :   Tuple {getElements(IndexerG<N>{}, arr)}
        {}

        /**
         * Initializes a new tuple from a const array.
         * @param arr The array to initialize tuple.
         */
        template <typename U = T, typename = typename std::enable_if<
                !std::is_reference<U>::value
            >::type >
        __host__ __device__ inline constexpr TupleN(const Pure<T> *arr) noexcept
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
        __host__ __device__ inline static constexpr auto getElements(Indexer<I...>, U *arr) noexcept
        -> Tuple
        {
            return {arr[I]...};
        }
};

namespace tuple
{
    /**
     * Retrieves and returns the value of the first leaf of a tuple.
     * @tparam T The type of tuple's first leaf member.
     * @param lead The first leaf member of a given tuple.
     * @return The head value of tuple.
     */
    template <typename T>
    __host__ __device__ inline constexpr const T& head(const TupleLeaf<0, T>& leaf) noexcept
    {
        return leaf.value;
    }

    /**
     * Removes the first element of the tuple and returns the rest.
     * @tparam I The list of tuple indeces to move to the new tuple.
     * @tparam T The list of types to move to the new tuple.
     * @param t The tuple to have its head removed.
     * @return The tail of given tuple.
     */
    template <size_t ...I, typename ...T, typename H>
    __host__ __device__ inline constexpr auto tail(const BaseTuple<Indexer<0, I...>, H, T...>& t) noexcept
    -> Tuple<T...>
    {
        return {static_cast<const T&>(get<I>(t))...};
    }

    /**#@+
     * Concatenates a list of tuples into a single one.
     * @param a The first tuple to concatenate.
     * @param b The second tuple to concatenate.
     * @param base The base tuple, which no other need to concatenate with.
     * @param tail The following tuples to concatenate.
     * @return A concatenated tuple of all others.
     */
    template <typename ...T>
    __host__ __device__ inline constexpr auto concat(const Tuple<T...>& base) noexcept
    -> Tuple<T...>
    {
        return base;
    }

    template <size_t ...I, size_t ...J, typename ...T, typename ...U, typename ...R>
    __host__ __device__ inline constexpr auto concat
        (   const BaseTuple<Indexer<I...>, T...>& a
        ,   const BaseTuple<Indexer<J...>, U...>& b
        ,   const R&... tail                            )
        noexcept
    -> decltype(concat(std::declval<Tuple<T..., U...>>(), std::declval<R>()...))
    {
        using M = Tuple<T..., U...>;
        return concat(M {static_cast<T>(get<I>(a))..., static_cast<U>(get<J>(b))...}, tail...);
    }
    /**#@-*/

    /**#@+
     * Performs a left fold, or reduction in the given tuple.
     * @tparam F The functor type to combine the values.
     * @tparam B The base and return fold value type.
     * @tparam T The tuple value types.
     * @param func The functor used to created the new elements.
     * @param base The base folding value.
     * @param t The tuple to fold.
     * @return The final value.
     */
    template <typename F, typename B>
    __host__ __device__ inline constexpr B foldl(F&& func, const B& base, const Tuple<>&)
    {
        return base;
    }

    template <typename F, typename B, typename ...T>
    __host__ __device__ inline constexpr B foldl(F&& func, const B& base, const Tuple<T...>& t)
    {
        return foldl(func, func(base, static_cast<B>(head(t))), tail(t));
    }
    /**#@-*/

    /**#@+
     * Performs a right fold, or reduction in the given tuple.
     * @tparam F The functor type to combine the values.
     * @tparam B The base and return fold value type.
     * @tparam T The tuple value types.
     * @param func The functor used to created the new elements.
     * @param base The base folding value.
     * @param t The tuple to fold.
     * @return The final value.
     */
    template <typename F, typename B>
    __host__ __device__ inline constexpr B foldr(F&& func, const B& base, const Tuple<>&)
    {
        return base;
    }

    template <typename F, typename B, typename ...T>
    __host__ __device__ inline constexpr B foldr(F&& func, const B& base, const Tuple<T...>& t)
    {
        return func(static_cast<B>(head(t)), foldr(func, base, tail(t)));
    }
    /**#@-*/

    /**#@+
     * Applies a left fold and returns all intermediate and final steps.
     * @tparam F The functor type to combine the values.
     * @tparam B The base and return fold value type.
     * @tparam I The tuple index counter.
     * @tparam T The tuple types.
     * @param func The functor used to created the new elements.
     * @param base The base folding value.
     * @param t The tuple to fold.
     * @return The intermediate and final values.
     */
    template <typename F, typename B>
    __host__ __device__ inline constexpr auto scanl(F&& func, const B& base, const BaseTuple<Indexer<>>&)
    -> Tuple<B>
    {
        return {base};
    }

    template <typename F, typename B, typename ...T, size_t ...I>
    __host__ __device__ inline constexpr auto scanl(F&& func, const B& base, const BaseTuple<Indexer<I...>, T...>& t)
    -> Tuple<B, Identity<B, I>...>
    {
        return {base, get<I>(scanl(func, func(base, static_cast<B>(head(t))), tail(t)))...};
    }
    /**#@-*/

    /**#@+
     * Applies a right fold and returns all intermediate and final steps.
     * @tparam F The functor type to combine the values.
     * @tparam B The base and return fold value type.
     * @tparam I The tuple index counter.
     * @tparam T The tuple types.
     * @param func The functor used to created the new elements.
     * @param base The base folding value.
     * @param t The tuple to fold.
     * @return The intermediate and final values.
     */
    template <typename F, typename B>
    __host__ __device__ inline constexpr auto scanr(F&& func, const B& base, const BaseTuple<Indexer<>>&)
    -> Tuple<B>
    {
        return {base};
    }

    template <typename F, typename B, typename ...T, size_t ...I>
    __host__ __device__ inline constexpr auto scanr(F&& func, const B& base, const BaseTuple<Indexer<I...>, T...>& t)
    -> Tuple<Identity<B, I>..., B>
    {
        return {get<I>(scanr(func, func(static_cast<B>(head(t)), base), tail(t)))..., base};
    }
    /**#@-*/

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

    /**
     * Creates a new tuple with elements calculated from the given functor and
     * the elements of input tuples occuring at the same position in both tuples.
     * @tparam F The functor type to use to construct the new tuple.
     * @tparam I The list of tuple indeces in the input tuples.
     * @tparam T The list of types in the first input tuple.
     * @tparam U The list of types in the second input tuple.
     * @param func The functor used to created the new elements.
     * @param a The first input tuple.
     * @param b The second input tuple.
     * @param The new tuple with the calculated elements.
     */
    template <typename F, size_t ...I, typename ...T, typename ...U>
    __host__ __device__ inline constexpr auto zipWith
        (   F&& func
        ,   const BaseTuple<Indexer<I...>, T...>& a
        ,   const BaseTuple<Indexer<I...>, U...>& b     )
    -> Tuple<T...>
    {
        return {func(static_cast<T>(get<I>(a)), static_cast<T>(get<I>(b)))...};
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

#endif