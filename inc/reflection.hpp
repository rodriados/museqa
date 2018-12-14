/** 
 * Multiple Sequence Alignment reflection header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Alexandr Poltavsky, Antony Polukhin, Rodrigo Siqueira
 */
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
#pragma once

#include <utility>
#include <cstddef>
#include <cstdint>

#if defined(__GNUC__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wnon-template-friend"
#endif

namespace reflection
{
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
     * Base for representing a tuple member and holding its value.
     * @tparam N The index of the member of the tuple.
     * @tparam T The type of the member of the tuple.
     * @since 0.1.1
     */
    template <size_t N, typename T>
    struct BaseMember
    {
        T value;    /// The value held by this tuple member.
    };

    /**#@+
     * The base struct for a type tuple.
     * @tparam I The indeces for the tuple members.
     * @tparam T The types of the tuple members.
     * @since 0.1.1
     */
    template <typename I, typename ...T>
    struct BaseTuple;

    template <size_t ...I, typename ...T>
    struct BaseTuple<std::index_sequence<I...>, T...> : BaseMember<I, T>...
    {
        static constexpr size_t size = sizeof...(I);    /// The size of the tuple.

        constexpr BaseTuple() noexcept = default;
        constexpr BaseTuple(BaseTuple&&) noexcept = default;
        constexpr BaseTuple(const BaseTuple&) noexcept = default;

        /**
         * This constructor sets every member with its corresponding value.
         * @param value The list of values for members.
         */
        constexpr BaseTuple(T... value) noexcept
        :   BaseMember<I, T> {value}... {}
    };

    template <>
    struct BaseTuple<std::index_sequence<>>
    {
        static constexpr size_t size = 0;   /// The size of the tuple.
    };
    /**#@-*/

    /*
     * Forward declaration of Tuple, so it can be used as a type for the
     * namespace's internal getter functions.
     */
    template <typename ...T>
    struct Tuple;

    namespace internal
    {
        /**
         * Chooses the requested member base and returns its value.
         * @param base The selected tuple member base.
         * @tparam N The requested member index.
         * @tparam T The member type to be matched.
         * @return The member's value.
         */
        template <size_t N, typename T>
        constexpr const T& t_get_impl(const BaseMember<N, T>& base) noexcept
        {
            return base.value;
        }

        /**
         * Retrieves the value held by a member base.
         * @param tuple The tuple in which value will be retrieved from.
         * @tparam N The requested member index.
         * @tparam T The member type to be matched.
         * @return The requested member's value.
         */
        template <size_t N, typename ...T>
        constexpr decltype(auto) t_get(const Tuple<T...>& tuple) noexcept
        {
            static_assert(N < Tuple<T...>::size, "Requested tuple index is out of bounds!");
            return t_get_impl<N>(tuple);
        }

        /**
         * Cleans a type from any reference, constness, volatile-ness or the like.
         * @tparam T The type to be cleaned.
         * @since 0.1.1
         */
        template <typename T>
        using clean = std::remove_cv_t<std::remove_reference_t<T>>;
    };

    /**
     * Tuple responsible for representing a struct to be translated.
     * @tparam T The tuple's list of member types.
     * @since 0.1.1
     */
    template <typename ...T>
    struct Tuple : BaseTuple<std::make_index_sequence<sizeof...(T)>, T...>
    {
        using BaseTuple<std::make_index_sequence<sizeof...(T)>, T...>::BaseTuple;

        /**
         * Gets value from member by index.
         * @tparam N The index of requested member.
         * @return The member's value.
         */
        template <size_t N>
        constexpr decltype(auto) get() noexcept
        {
            return internal::t_get<N>(*this);
        }
    };

    /**
     * The type of a tuple element.
     * @tparam I The index of tuple element.
     * @tparam T The target tuple.
     * @since 0.1.1
     */
    template <size_t I, typename T>
    using TupleElement = std::remove_reference<decltype(internal::t_get<I>(std::declval<T>()))>;

    namespace internal
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
                return std::remove_all_extents_t<U>();
            }
        };

        // This specialization avoids multiple definition errors.
        template <typename T, typename U, size_t N>
        struct TagDef<T, U, N, true> {};
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

        template <typename U, size_t M, size_t = sizeof(loophole(internal::Tag<T, M>{}))>
        static auto ins(int) -> char;
        /**#@-*/

        /**
         * The casting operator function helps in the type detection of struct members.
         * @return Unknown.
         */
        template <typename U, size_t = sizeof(internal::TagDef<T, U, N, sizeof(ins<U, N>(0)) == sizeof(char)>)>
        constexpr operator U&() const noexcept;
    };

    namespace internal
    {
        /**#@+
         * This struct is a helper for creating an aligned tuple. This is useful for
         * retrieving offsets out of tuples.
         * @tparam T The list of tuple's types.
         * @since 0.1.1
         */
        template <class T>
        struct AlignedStorageTuple;

        template <typename ...T>
        struct AlignedStorageTuple<Tuple<T...>>
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
    using AlignedTuple = typename internal::AlignedStorageTuple<T>::type;

    /**
     * This type turns a data structure into a tuple.
     * @tparam T The structure to be transformed.
     * @since 0.1.1
     */
    template <typename T>
    using LoopholeTuple = typename internal::LoopholeTypeList
        <   internal::clean<T>
        ,   std::make_index_sequence<internal::count<internal::clean<T>>(0)> >
        ::type;
};

/**
 * Applies reflection over a data structure, thus allowing us to automagically get
 * information about the structure during compile- and run-times.
 * @tparam T The data structure to be introspected.
 * @since 0.1.1
 */
template <typename T>
class Reflection
{
    private:
        /**
         * Cleaning the type to be reflected.
         * @since 0.1.1
         */
        using Type = reflection::internal::clean<T>;

    public:
        /**
         * The tuple aligned to reflected type.
         * @since 0.1.1
         */
        using Tuple = reflection::LoopholeTuple<Type>;

        static_assert(!std::is_union<T>::value, "It is forbidden to reflect unions!");
        static_assert(sizeof(Type) == sizeof(Tuple), "Member sequence is not compatible!");
        static_assert(alignof(Type) == alignof(Tuple), "Member sequence is not compatible!");

        /**
         * Retrieves the offset of a member in the data structure by its index.
         * @tparam N The index of required member.
         * @return The member offset.
         */
        template <size_t N>
        static constexpr ptrdiff_t getOffset() noexcept
        {
            namespace r = reflection;            
            constexpr r::AlignedTuple<Tuple> l {};            
            return &r::internal::t_get<N>(l).storage[0] - &r::internal::t_get<0>(l).storage[0];
        }

        /**
         * Retrieves the number of members of the given data structure.
         * @return The number of structure's members.
         */
        static constexpr size_t getSize() noexcept
        {
            return reflection::internal::count<Type>(0);
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
            return Tuple(values...);
        }
};

#if defined(__GNUC__)
  #pragma GCC diagnostic pop
#endif

#endif