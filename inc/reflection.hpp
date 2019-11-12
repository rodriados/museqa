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

#include <utils.hpp>
#include <tuple.hpp>

#if defined(__GNUC__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wnon-template-friend"
#endif

#ifndef use_loophole
  #if __cplusplus >= 201402L && !defined(avoid_loophole)
    #define use_loophole 1
  #else
    #define use_loophole 0
  #endif
#endif

namespace internal
{
    #if use_loophole
        namespace reflection
        {
            /**
             * Generates friend declarations and tags type for overload resolution.
             * @tparam T The object type to be scanned.
             * @tparam N The index of requested object member.
             * @since 0.1.1
             */
            template <typename T, size_t N>
            struct tag
            {
                friend auto loop(tag<T, N>);
            };

            /**#@+
             * Defines the friend function that automagically returns an object's
             * default constructor parameter types.
             * @since 0.1.1
             */
            template <typename T, typename U, size_t N, bool B>
            struct tagdef
            {
                /**
                 * This function automagically returns an object's default constructor
                 * parameter types, and thus, its internal member's types.
                 * @return An instance of the resolved type.
                 */
                friend auto loop(tag<T, N>)
                {
                    return typename std::remove_all_extents<U>::type {};
                }
            };

            // This specialization avoids multiple definition errors.
            template <typename T, typename U, size_t N>
            struct tagdef<T, U, N, true>
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
            struct loophole
            {
                /**#@+
                 * This method is responsible for doing the type detection using the constexpr
                 * friend function and SFINAE. The return value of this function is never used
                 * or known, we only care about its type.
                 * @return Unknown.
                 */
                template <typename U, size_t M>
                static auto ins(...) -> size_t;

                template <typename U, size_t M, size_t = sizeof(loop(reflection::tag<T, M>{}))>
                static auto ins(int) -> char;
                /**#@-*/

                /**
                 * The casting operator identifies the type required by the object's default constructor.
                 * @return Unknown.
                 */
                template <
                        typename U
                    ,   size_t = sizeof(reflection::tagdef<T, U, N, sizeof(ins<U, N>(0)) == sizeof(char)>)
                    >
                constexpr operator U&() const noexcept;
            };

            /**#@+
             * Generates the tuple corresponding to given type using the loophole.
             * @tparam T The object to be scanned.
             * @tparam I The object's number of members.
             * @since 0.1.1
             */
            template <class I, class T>
            struct loophole_invoker;

            template <size_t ...I, typename T>
            struct loophole_invoker<indexer<I...>, T> : ::tuple<decltype(T {loophole<T, I> {}...}, 0)>
            {
                using type = ::tuple<decltype(loop(tag<T, I> {}))...>;
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
            constexpr auto count(int) -> decltype(T {loophole<T, N> {}...}, 0)
            {
                return count<T, N..., sizeof...(N)>(0);
            }
            /**#@-*/
        }
    #endif

    namespace reflection
    {
        #if use_loophole
            /**
             * This type uses C++14 loophole to create a tuple corresponding
             * to the object's fields.
             * @tparam T The object to reflect on.
             * @since 0.1.1
             */
            template <typename T>
            using loophole_tuple = typename loophole_invoker<indexer_g<count<T>(0)>, T>::type;
        #else
            /**
             * Informs whether loophole is needed to reflect on the object.
             * @tparam T The object to reflect on.
             * @since 0.1.1
             */
            template <typename T>
            using loophole_tuple = void;
        #endif

        /**#@+
         * Helper of creating tuple corresponding to reflected object.
         * @tparam T The object to reflect on.
         * @tparam B Does the object inherit Reflector?
         * @since 0.1.1
         */
        template <typename T, bool B, typename = void>
        struct mirror_tuple;

        template <typename T>
        struct mirror_tuple<T, false, typename std::enable_if<
                std::is_trivial<T>::value
            &&  std::is_standard_layout<T>::value
            &&  use_loophole
            >::type >
        {
            using type = loophole_tuple<T>;
        };

        template <typename T>
        struct mirror_tuple<T, true, typename std::enable_if<
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
        static constexpr auto aligned(::tuple<T...>) noexcept
        -> ::tuple<storage<sizeof(T), alignof(T)>...>;

        /**
         * Helper for creating a tuple with references to object values.
         * @tparam T The list of object's types.
         * @since 0.1.1
         */
        template <typename ...T>
        static constexpr auto reference(::tuple<T...>) noexcept
        -> ::tuple<T&...>;
    }
}

/**
 * Indicates whether a non-trivial object is reflectible.
 * @since 0.1.1
 */
class reflector
{
    public:
        /**
         * If reflection is manually generated, then this must be a tuple similar
         * and equivalent to the object being reflected.
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
        static constexpr auto reflect(T&...) noexcept
        -> decltype(utils::concat(std::declval<ntuple<
                typename std::remove_all_extents<T>::type
            ,   utils::max(std::extent<T>::value, 1ul)
            >>()...));
};

/**
 * Reflects on the object and creates a tuple corresponding to its properties.
 * @tparam T The object to reflect on.
 * @since 0.1.1
 */
template <typename T>
using reflection_tuple = typename internal::reflection::mirror_tuple<
        T
    ,   std::is_base_of<reflector, T>::value
    >::type;

/**
 * This type creates a tuple in which offsets are aligned to the those of the
 * base data object.
 * @tpatam T The base tuple for an object's properties.
 * @since 0.1.1
 */
template <typename T>
using aligned_tuple = decltype(internal::reflection::aligned(std::declval<reflection_tuple<T>>()));

/**
 * This type creates a tuple with references to the object's properties.
 * @tpatam T The base tuple for an object's properties.
 * @since 0.1.1
 */
template <typename T>
using reference_tuple = decltype(internal::reflection::reference(std::declval<reflection_tuple<T>>()));

/**
 * Applies reflection over a data object, thus allowing us to automagically get
 * information about the object during compile- and run-times.
 * @tparam T The data object to be introspected.
 * @since 0.1.1
 */
template <typename T>
class reflection : public reference_tuple<T>
{
    static_assert(!std::is_union<T>::value, "it is forbidden to reflect over unions");
    static_assert(std::is_class<T>::value, "the reflected object must be a class or struct");
    static_assert(sizeof(T) == sizeof(reflection_tuple<T>), "reflection tuple is not compatible");
    static_assert(alignof(T) == alignof(reflection_tuple<T>), "reflection tuple is not compatible");

    protected:
        using underlying_tuple = reference_tuple<T>;    /// The underlying tuple type.

    public:
        inline reflection() = delete;
        inline reflection(const reflection&) = default;
        inline reflection(reflection&&) = delete;

        /**
         * Gathers references to an instance of the reflected object.
         * @param obj The object instance to get references from.
         */
        __host__ __device__ inline reflection(T& obj) noexcept
        :   underlying_tuple {extract(*this, indexer_g<count()> {}, obj)}
        {}

        using underlying_tuple::operator=;

        /**
         * Retrieves the number of members of the given data object.
         * @return The number of object's members.
         */
        __host__ __device__ inline static constexpr auto count() noexcept -> size_t
        {
            return underlying_tuple::count;
        }

        /**
         * Retrieves the offset of a member in the data object by its index.
         * @tparam N The index of required member.
         * @return The member offset.
         */
        template <size_t N>
        __host__ __device__ inline static constexpr auto offset() noexcept -> ptrdiff_t
        {
            return offset<N>(aligned_tuple<T> {});
        }

    private:
        /**
         * Retrieves the offset of a member in the data object by its index.
         * @tparam N The index of required member.
         * @param t An object's corresponding alignment tuple instance.
         * @return The member offset.
         */
        template <size_t N>
        __host__ __device__ inline static constexpr auto offset(aligned_tuple<T> tp) noexcept -> ptrdiff_t
        {
            return &tp.template get<N>().storage[0] - &tp.template get<0>().storage[0];
        }

        /**
         * Retrieves references to an instance of the reflected object's properties.
         * @tparam U The list of types on reflected object.
         * @tparam I The types index sequence.
         * @param obj The object instance to gather references from.
         * @return The new reference tuple instance.
         */
        template <typename ...U, size_t ...I>
        __host__ __device__ inline static auto extract(tuple<U...>&, indexer<I...>, T& obj) noexcept
        -> underlying_tuple
        {
            return {reinterpret_cast<U>(*(reinterpret_cast<char *>(&obj) + offset<I>()))...};
        }
};

#if defined(__GNUC__)
  #pragma GCC diagnostic pop
#endif

#endif