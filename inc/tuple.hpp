/** 
 * Multiple Sequence Alignment tuple header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef TUPLE_HPP_INCLUDED
#define TUPLE_HPP_INCLUDED

#include <utility>

#include "utils.hpp"

/*
 * Forward declaration of base tuple struct, so it can be used as a type
 * before its actual definition.
 */
template <typename I, typename ...T>
struct BaseTuple;

/*
 * Forward declaration of a tuple, so it can be used as a type in the
 * namespace's internal getter functions.
 */
template <typename ...T>
struct Tuple;

namespace tuple
{
    /**
     * Base for representing a tuple member and holding its value.
     * @tparam I The index of the member of the tuple.
     * @tparam T The type of the member of the tuple.
     * @since 0.1.1
     */
    template <size_t I, typename T>
    struct BaseMember
    {
        T value = {};   /// The value held by this tuple member.
    };

    /**
     * Retrieves the requested member base and returns its value.
     * @param base The selected tuple member base.
     * @tparam I The requested member index.
     * @tparam T The member type to be matched.
     * @return The member's value.
     */
    template <size_t I, typename T>
    inline constexpr const T& retrieve(const BaseMember<I, T>& base) noexcept
    {
        return base.value;
    }

    /**
     * Retrieves the value held by a member base.
     * @param tuple The tuple in which value will be retrieved from.
     * @tparam IN The requested member index.
     * @tparam T The member type to be matched.
     * @return The requested member's value.
     */
    template <size_t I, typename ...T>
    inline constexpr decltype(auto) get(const Tuple<T...>& tuple) noexcept
    {
        static_assert(I < sizeof...(T), "requested tuple index is out of bounds!");
        return retrieve<I>(tuple);
    }
};

/**#@+
 * The base struct for a type tuple.
 * @tparam I The indeces for the tuple members.
 * @tparam T The types of the tuple members.
 * @since 0.1.1
 */
template <size_t ...I, typename ...T>
struct BaseTuple<std::index_sequence<I...>, T...> : public tuple::BaseMember<I, T>...
{
    constexpr BaseTuple() noexcept = default;
    constexpr BaseTuple(const BaseTuple&) noexcept = default;
    constexpr BaseTuple(BaseTuple&&) noexcept = default;

    /**
     * This constructor sets every base member with its corresponding value.
     * @param value The list of values for members.
     */
    inline constexpr BaseTuple(T... value) noexcept
    :   tuple::BaseMember<I, T> {value}...
    {}

    static constexpr size_t size = sizeof...(I);    /// The size of the tuple.
};

template <>
struct BaseTuple<std::index_sequence<>>
{
    static constexpr size_t size = 0;   /// The size of the tuple.
};
/**#@-*/

/**
 * A tuple is responsible for holding a list of elements of possible different
 * types with a known number of elements.
 * @tparam T The tuple's list of member types.
 * @since 0.1.1
 */
template <typename ...T>
struct Tuple : public BaseTuple<std::make_index_sequence<sizeof...(T)>, T...>
{
    /**
     * Gets value from member by index.
     * @tparam I The index of requested member.
     * @return The member's value.
     */
    template <size_t I>
    constexpr decltype(auto) get() const noexcept
    {
        return tuple::get<I>(*this);
    }

    using BaseTuple<std::make_index_sequence<sizeof...(T)>, T...>::BaseTuple;
};

namespace tuple
{
    /**#@+
     * Repeats a type so it's easier to create tuples with repeated types.
     * @tparam I The number of times the type shall repeat.
     * @tparam T The type to be repeated.
     * @since 0.1.1
     */
    template <typename I, typename T>
    struct Repeater;

    template <size_t ...I, typename T>
    struct Repeater<std::index_sequence<I...>, T>
    {
        using type = Tuple<Identity<T, I>...>;
    };
    /**#@-*/
};

/**
 * Creates a tuple with many instances of a single type.
 * @tparam T The type to create tuple from.
 * @tparam N The number of types the type shall repeat.
 * @since 0.1.1
 */
template <typename T, size_t N>
using TupleN = typename tuple::Repeater<std::make_index_sequence<N>, T>::type;

/**
 * The type of a tuple element.
 * @tparam I The index of tuple element.
 * @tparam T The target tuple.
 * @since 0.1.1
 */
template <size_t I, typename T>
using TupleElement = typename std::remove_reference<decltype(tuple::get<I>(T{}))>::type;

#endif