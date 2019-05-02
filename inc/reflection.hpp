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
 * Note: CWG agreed that such techniques should be ill-formed, although the mechanism for
 * prohibiting them is as yet undetermined.
 */
#include <cstddef>
#include <cstdint>
#include <utility>

#include "utils.hpp"
#include "tuple.hpp"

#if defined(__GNUC__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wnon-template-friend"
#endif

#ifndef reflection_use_loophole
  #if __cplusplus >= 201402L && !defined(msa_avoid_loophole)
    #define reflection_use_loophole 1
  #else
    #define reflection_use_loophole 0
  #endif
#endif

namespace reflection
{
#if reflection_use_loophole
    namespace detail
    {
        /**
         * Generates friend declarations and tags type for overload resolution.
         * @tparam T The object type to be scanned.
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

        /**
         * Templated conversion operator that triggers instantiations. The use of `sizeof`
         * here is important as it seems to be more reliable. An template argument `U` is
         * provided so the template arguments do not get "cached" (although it is not known
         * whether they are really cached or not).
         * @tparam T The object type to be scanned.
         * @tparam N The counter of members.
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
             * The casting operator identifies the type required by the object's default constructor.
             * @return Unknown.
             */
            template <typename U, size_t = sizeof(detail::TagDef<T, U, N, sizeof(ins<U, N>(0)) == sizeof(char)>)>
            constexpr operator U&() const noexcept;
        };

        /**#@+
         * Generates the tuple corresponding to given type using the loophole.
         * @tparam T The object to be scanned.
         * @tparam I The object's number of members.
         * @since 0.1.1
         */
        template <class I, class T>
        struct LoopholeInvoker;

        template <size_t ...I, typename T>
        struct LoopholeInvoker<Index<I...>, T> : Tuple<decltype(T{Loophole<T, I>{}...}, 0)>
        {
            using type = Tuple<decltype(loophole(Tag<T, I>{}))...>;
        };
        /**#@-*/

        /**#@+
         * Automagically counts the number of members of a data object.
         * @tparam T The object to be scanned.
         * @tparam N The number of members.
         * @return The number of members in the data object.
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
#endif

#if reflection_use_loophole
    /**
     * This type uses C++14 loophole to create a tuple corresponding
     * to the object's fields.
     * @tparam T The object to reflect on.
     * @since 0.1.1
     */
    template <typename T>
    using LoopholeTuple = typename detail::LoopholeInvoker<
            typename IndexGen<detail::count<T>(0)>::type, T
        >::type;
#else
    /**
     * Informs whether loophole is needed to reflect on the object.
     * @tparam T The object to reflect on.
     * @since 0.1.1
     */
    template <typename T>
    using LoopholeTuple = void;
#endif

    /**#@+
     * Helper of creating tuple corresponding to reflected object.
     * @tparam T The object to reflect on.
     * @tparam B Does the object inherit Reflector?
     * @since 0.1.1
     */
    template <typename T, bool B, typename = void>
    struct ReflectionTuple;

    template <typename T>
    struct ReflectionTuple<T, false, typename std::enable_if<
            std::is_trivial<T>::value &&
            std::is_standard_layout<T>::value &&
            reflection_use_loophole
        >::type >
    {
        using type = LoopholeTuple<T>;
    };

    template <typename T>
    struct ReflectionTuple<T, true, typename std::enable_if<
            !std::is_same<typename T::reflex, void>::value
        >::type >
    {
        using type = typename T::reflex;
    };
    /**#@-*/

    /**
     * Helper for creating an aligned tuple. This is useful for retrieving
     * correct and reliable offsets out of tuples.
     * @tparam T The list of object's types.
     * @since 0.1.1
     */
    template <typename ...T>
    static constexpr auto aligned(Tuple<T...>) noexcept
    -> Tuple<AlignedStorage<sizeof(T), alignof(T)>...>;

    /**
     * Helper for creating a tuple with references to object values.
     * @tparam T The list of object's types.
     * @since 0.1.1
     */
    template <typename ...T>
    static constexpr auto reference(Tuple<T...>) noexcept
    -> Tuple<T&...>;
};

/**
 * Indicates whether a non-trivial object is reflectible.
 * @since 0.1.1
 */
class Reflector
{
    public:
        /**
         * If reflection is manually generated, then this must be the tuple
         * equivalent to the object being reflected.
         * @since 0.1.1
         */
        using reflex = void;

    protected:
        /**
         * Creates the reflection tuple from the list of object's properties.
         * @tparam T The object's properties types.
         * @param (ignored) The object's properties references.
         * @return The reflection tuple.
         */
        template <typename ...T>
        static constexpr auto reflect(const T&...) noexcept
        -> Tuple<typename std::remove_cv<T>::type...>;
};

/**
 * Reflects on the object and creates a tuple corresponding to its properties.
 * @tparam T The object to reflect on.
 * @since 0.1.1
 */
template <typename T>
using ReflectionTuple = typename reflection::ReflectionTuple<
        T, std::is_base_of<Reflector, T>::value
    >::type;

/**
 * This type creates a tuple in which offsets are aligned to the those of the
 * base data object.
 * @tpatam T The base tuple for an object's properties.
 * @since 0.1.1
 */
template <typename T>
using AlignedTuple = decltype(reflection::aligned(ReflectionTuple<T>{}));

/**
 * This type creates a tuple with references to the object's properties.
 * @tpatam T The base tuple for an object's properties.
 * @since 0.1.1
 */
template <typename T>
using ReferenceTuple = decltype(reflection::reference(ReflectionTuple<T>{}));

/**
 * Applies reflection over a data object, thus allowing us to automagically get
 * information about the object during compile- and run-times.
 * @tparam T The data object to be introspected.
 * @since 0.1.1
 */
template <typename T>
class Reflection : public ReferenceTuple<T>
{
    public:
        Reflection() = delete;
        Reflection(const Reflection&) = default;
        Reflection(Reflection&&) = delete;

        /**
         * Gathers references to an instance of the reflected object.
         * @param obj The object instance to get references from.
         */
        __host__ __device__ inline Reflection(T& obj) noexcept
        :   ReferenceTuple<T> {getReference(ReflectionTuple<T> {}, IndexGen<getSize()> {}, obj)}
        {}

        Reflection& operator=(const Reflection&) = delete;
        Reflection& operator=(Reflection&&) = delete;

        /**
         * Retrieves the offset of a member in the data object by its index.
         * @tparam N The index of required member.
         * @return The member offset.
         */
        template <size_t N>
        static constexpr ptrdiff_t getOffset() noexcept
        {
            return getOffset<N>(AlignedTuple<T> {});
        }

        /**
         * Retrieves the number of members of the given data object.
         * @return The number of object's members.
         */
        static constexpr size_t getSize() noexcept
        {
            return ReflectionTuple<T>::count;
        }

    private:
        /**
         * Retrieves the offset of a member in the data object by its index.
         * @tparam N The index of required member.
         * @param t An object's corresponding alignment tuple instance.
         * @return The member offset.
         */
        template <size_t N>
        static constexpr ptrdiff_t getOffset(AlignedTuple<T> t) noexcept
        {
            return &t.template get<N>().storage[0] - &t.template get<0>().storage[0];
        }

        /**
         * Retrieves references to an instance of the reflected object's properties.
         * @tparam U The list of types on reflected object.
         * @tparam I The types index sequence.
         * @param obj The object instance to gather references from.
         * @return The new reference tuple instance.
         */
        template <typename ...U, size_t ...I>
        __host__ __device__ inline static ReferenceTuple<T> getReference(Tuple<U...>, Index<I...>, T& obj)
        {
            return {*reinterpret_cast<U*>(reinterpret_cast<char *>(&obj) + getOffset<I>())...};
        }

    static_assert(!std::is_union<T>::value, "it is forbidden to reflect over unions!");
    static_assert(std::is_class<T>::value, "the reflected object must be a class or struct!");
    static_assert(sizeof(T) == sizeof(ReflectionTuple<T>), "reflection tuple is not compatible!");
    static_assert(alignof(T) == alignof(ReflectionTuple<T>), "reflection tuple is not compatible!");
};

#if defined(__GNUC__)
  #pragma GCC diagnostic pop
#endif

#endif