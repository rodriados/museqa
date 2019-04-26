/** 
 * Multiple Sequence Alignment reflection header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Alexandr Poltavsky, Antony Polukhin, Rodrigo Siqueira
 */
#pragma once

#ifndef REFLECTION_HPP_INCLUDED
#define REFLECTION_HPP_INCLUDED

/*
 * The Great Type Loophole (C++14)
 * Initial implementation by Alexandr Poltavsky, http://alexpolt.github.io
 * With participation of Antony Polukhin, http://github.com/apolukhin
 *
 * The Great Type Loophole is a technique that allows to exchange type information with template
 * instantiations. Basically you can assign and read type information during compile time.
 * Here it is used to detect data members of a data type. I described it for the first time in
 * this blog post http://alexpolt.github.io/type-loophole.html .
 *
 * This technique exploits the http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_active.html#2118
 * CWG 2118. Stateful metaprogramming via friend injection
 * Note: CWG agreed that such techniques should be ill-formed, although the mechanism for prohibiting them
 * is as yet undetermined.
 */
#include <utility>
#include <cstddef>
#include <cstdint>

#include "utils.hpp"
#include "tuple.hpp"

#if defined(__GNUC__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wnon-template-friend"
#endif

namespace reflection
{
    namespace detail
    {
        /**
         * Generates friend declarations and helps with overload resolution.
         * @tparam T The object type to be scaned.
         * @tparam N The index of requested object member.
         * @since 0.1.1
         */
        template <typename T, size_t N>
        struct Tag
        {
            friend auto loophole(Tag<T, N>);
        };

        /**#@+
         * Defines the friend function that automagically returns member's types.
         * @since 0.1.1
         */
        template <typename T, typename U, size_t N, bool B>
        struct TagDef
        {
            /**
             * This function automagically returns object member's types.
             * @return An instance of the resolved type.
             */
            friend auto loophole(Tag<T, N>)
            {
                return typename std::remove_all_extents<U>::type {};
            }
        };

        // This specialization avoids multiple definition errors.
        template <typename T, typename U, size_t N>
        struct TagDef<T, U, N, true>
        {};
        /**#@-*/
    };

    /**
     * Templated conversion operator that triggers instantiations. The use of `sizeof`
     * here is important as it seems to be more reliable. An template argument `U` is
     * provided so the template arguments do not get "cached" (although it is not known
     * whether they are really cached or not).
     * @tparam T The object type to be scaned.
     * @tparam N The number of members.
     * @since 0.1.1
     */
    template <typename T, size_t N>
    struct Loophole
    {
        /**#@+
         * This method is responsible for doing the type detection using the constexpr
         * friend function and SFINAE. The return value of this function is never used
         * or known, we only care about its type.
         * @return Unknown.
         */
        template <typename U, size_t M>
        static auto ins(...) -> size_t;

        template <typename U, size_t M, size_t = sizeof(loophole(detail::Tag<T, M>{}))>
        static auto ins(int) -> char;
        /**#@-*/

        /**
         * The casting operator function helps in the type detection of struct members.
         * @return Unknown.
         */
        template <typename U, size_t = sizeof(detail::TagDef<T, U, N, sizeof(ins<U, N>(0)) == sizeof(char)>)>
        constexpr operator U&() const noexcept;
    };

    namespace detail
    {
        /**#@+
         * This struct is a helper for creating an aligned tuple. This is useful for
         * retrieving offsets out of tuples.
         * @tparam T The list of tuple's types.
         * @since 0.1.1
         */
        template <class T>
        struct AlignedTuple;

        template <typename ...T>
        struct AlignedTuple<Tuple<T...>>
        {
            using type = Tuple<AlignedStorage<sizeof(T), alignof(T)>...>;
        };
        /**#@-*/

        /**#@+
         * This struct is a helper to turn a data structure into a tuple.
         * @tparam T The structure to be scaned.
         * @tparam I The structure's number of members.
         * @since 0.1.1
         */
        template <class T, class U>
        struct LoopholeTypeList;

        template <typename T, size_t ...I>
        struct LoopholeTypeList<T, std::index_sequence<I...>> : Tuple<decltype(T{Loophole<T, I>{}...}, 0)>
        {
            using type = Tuple<decltype(loophole(Tag<T, I>{}))...>;
        };
        /**#@-*/

        /**#@+
         * Automagically counts the number of members of a data structure.
         * @tparam T The structure to be scanned.
         * @tparam N The number of members.
         * @return The number of members in the data structure.
         */
        template <typename T, size_t ...N>
        constexpr size_t count(...)
        {
            return sizeof...(N) - 1;
        }

        template <typename T, size_t ...N>
        constexpr auto count(int) -> decltype(T{Loophole<T, N>{}...}, 0)
        {
            return count<T, N..., sizeof...(N)>(0);
        }
        /**#@-*/
    };

    /**
     * This type creates a tuple in which offsets are aligned to the those of the
     * base data structure.
     * @tparam T The base structure for the tuple.
     * @since 0.1.1
     */
    template <typename T>
    using AlignedTuple = typename detail::AlignedTuple<T>::type;

    /**
     * This type turns a data structure into a tuple.
     * @tparam T The structure to be transformed.
     * @since 0.1.1
     */
    template <typename T>
    using LoopholeTuple = typename detail::LoopholeTypeList<
            T, std::make_index_sequence<detail::count<T>(0)>
        >::type;
};

/**
 * Applies reflection over a data structure, thus allowing us to automagically get
 * information about the structure during compile- and run-times.
 * @tparam T The data structure to be introspected.
 * @since 0.1.1
 */
template <typename T>
struct Reflection
{
    /**
     * The tuple aligned to reflected type.
     * @since 0.1.1
     */
    using Tuple = reflection::LoopholeTuple<Base<T>>;

    static_assert(!std::is_union<T>::value, "it is forbidden to reflect over unions!");
    static_assert(sizeof(Base<T>) == sizeof(Tuple), "member sequence is not compatible!");
    static_assert(alignof(Base<T>) == alignof(Tuple), "member sequence is not compatible!");

    /**
     * Retrieves the offset of a member in the data structure by its index.
     * @tparam N The index of required member.
     * @return The member offset.
     */
    template <size_t N>
    static constexpr ptrdiff_t getOffset() noexcept
    {
        constexpr reflection::AlignedTuple<Tuple> t {};
        return &t.template get<N>().storage[0] - &t.template get<0>().storage[0];
    }

    /**
     * Retrieves the number of members of the given data structure.
     * @return The number of structure's members.
     */
    static constexpr size_t getSize() noexcept
    {
        return reflection::detail::count<Base<T>>(0);
    }

    /**
     * Creates a new tuple instance based on the reflected type.
     * @tparam U The list of types to create the new instance.
     * @param values The list of values.
     * @return The new tuple instance.
     */
    template <typename ...U>
    static constexpr Tuple newInstance(U... values)
    {
        return Tuple {values...};
    }
};

#if defined(__GNUC__)
  #pragma GCC diagnostic pop
#endif

#endif