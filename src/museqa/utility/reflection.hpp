/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Reflection implementation for simple data structures.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Alexandr Poltavsky, Antony Polukhin, Rodrigo Siqueira
 */
#pragma once

/*
 * The Great Type Loophole (C++14)
 * Initial implementation by Alexandr Poltavsky, http://alexpolt.github.io
 * With participation of Antony Polukhin, http://github.com/apolukhin
 *
 * The Great Type Loophole is a technique that allows to exchange type information
 * with template instantiations. Basically you can assign and read type information
 * during compile time. Here it is used to detect data members of a data type. I
 * described it for the first time in this blog post http://alexpolt.github.io/type-loophole.html .
 *
 * This technique exploits the http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_active.html#2118
 * CWG 2118. Stateful metaprogramming via friend injection
 * Note: CWG agreed that such techniques should be ill-formed, although the mechanism
 * for prohibiting them is as yet undetermined.
 */
#include <cstddef>
#include <cstdint>
#include <utility>

#include <museqa/utility.hpp>
#include <museqa/utility/tuple.hpp>
#include <museqa/environment.h>

#if !defined(MUSEQA_AVOID_REFLECTION)

/*
 * As we are exploiting some "obscure" behaviors of the language, and using some
 * tricks that upset compilers, we need to disable some warnings in order to force
 * the compilation to take place without any problems.
 */
MUSEQA_DISABLE_NVCC_WARNING_BEGIN(1301)
MUSEQA_DISABLE_GCC_WARNING_BEGIN("-Wnon-template-friend")

MUSEQA_BEGIN_NAMESPACE

namespace utility
{
    /**
     * Reflects over the target data type, that is it extracts information about
     * the member properties of the target type.
     * @tparam T The target data type to be introspected.
     * @since 1.0
     */
    template <typename T>
    class reflector_t;

    /**
     * Extracts and manages references to each member property of the target type,
     * thus enumerating each of the target type's property members and allowing
     * them to be directly accessed or updated.
     * @tparam T The target data type to be introspected.
     * @since 1.0
     */
    template <typename T>
    class reflection_t : public reflector_t<T>::reference_tuple_t
    {
        protected:
            typedef utility::reflector_t<T> reflector_t;
            typedef typename reflector_t::reference_tuple_t underlying_tuple_t;

        public:
            __host__ __device__ inline reflection_t() noexcept = delete;
            __host__ __device__ inline reflection_t(const reflection_t&) noexcept = default;
            __host__ __device__ inline reflection_t(reflection_t&&) noexcept = default;

            /**
             * Reflects over an instance and gathers refereces to its members.
             * @param target The target instance to get references from.
             */
            __host__ __device__ inline reflection_t(T& target) noexcept
              : underlying_tuple_t {extract(*this, target)}
            {}

            __host__ __device__ inline reflection_t& operator=(const reflection_t&) = default;
            __host__ __device__ inline reflection_t& operator=(reflection_t&&) = default;

        private:
            /**
             * Retrieves references to the properties of a reflected instance.
             * @tparam U The list of property member types.
             * @tparam I The member types index sequence.
             * @param target The reflected instance to gather references from.
             * @return The new reference tuple instance.
             */
            template <typename ...U, size_t ...I>
            __host__ __device__ inline static auto extract(
                tuple_t<identity_t<std::index_sequence<I...>>, U...>&, T& target
            ) noexcept -> underlying_tuple_t
            {
                return {reflector_t::template member<I>(target)...};
            }
    };

    namespace detail
    {
        /**
         * Tags a member property type to an index for overload resolution.
         * @tparam T The target type for reflection processing.
         * @tparam N The index of extracted property member type.
         * @since 1.0
         */
        template <typename T, size_t N>
        struct tag_t
        {
            friend auto latch(tag_t<T, N>) noexcept;
        };

        /**
         * Injects a friend function to couple a property type to its index.
         * @tparam T The target type for reflection processing.
         * @tparam U The extracted property type.
         * @param N The index of extracted property member type.
         * @since 1.0
         */
        template <typename T, typename U, size_t N>
        struct injector_t
        {
            /**
             * Binds the extracted member type to its index within the target reflection
             * type. This function does not aim to have its concrete return value
             * used, but only its return type.
             * @return The extracted type bound to the member index.
             */
            friend inline auto latch(tag_t<T, N>) noexcept
            {
                return typename std::remove_all_extents<U>::type {};
            }
        };

        /**
         * Decoy type responsible for pretending to be a type instance required
         * to build the target reflection type and latching the required type into
         * the injector, so that it can be retrieved later on.
         * @tparam T The target type for reflection processing.
         * @tparam N The index of property member type to extract.
         * @since 1.0
         */
        template <typename T, size_t N>
        struct decoy_t
        {
            /**
             * Injects the extracted member type into a latch if it has not yet
             * been previously done so.
             * @tparam U The extracted property member type.
             * @tparam M The index of the property member being processed.
             */
            template <typename U, size_t M>
            static constexpr auto inject(...) -> injector_t<T, U, M>;

            /**
             * Validates whether the type member being processed has already been
             * reflected over. If yes, avoids latch redeclaration.
             * @tparam M The index of the property member being processed.
             */
            template <typename, size_t M>
            static constexpr auto inject(int) -> decltype(latch(tag_t<T, M>()));

            /**
             * Morphs the decoy into the required type for constructing the target
             * reflection type and injects it into the type latch.
             * @tparam U The type to morph the decoy into.
             */
            template <typename U, size_t = sizeof(inject<U, N>(0))>
            constexpr operator U&() const noexcept;
        };

        /**#@+
         * Recursively counts the number of property members in the target type
         * of a reflection processing.
         * @tparam T The target type for reflection processing.
         * @return The total number of members within the target type.
         */
        template <typename T, size_t ...I>
        inline constexpr auto count(...) noexcept
        -> size_t { return sizeof...(I) - 1; }

        template <typename T, size_t ...I, size_t = sizeof(T{decoy_t<T, I>()...})>
        inline constexpr auto count(int) noexcept
        -> size_t { return count<T, I..., sizeof...(I)>(0); }
        /**#@-*/

        /**
         * Extracts the types of the property members within the target reflection
         * object type into an instantiable tuple.
         * @tparam T The target type for reflection processing.
         * @return The tuple of extracted types.
         */
        template <typename T, size_t ...I, typename = decltype(T{decoy_t<T, I>()...})>
        inline constexpr auto loophole(std::index_sequence<I...>) noexcept
        -> tuple_t<decltype(latch(tag_t<T, I>()))...>;

        /**
         * Retrieves the tuple of extracted property member types of the given target
         * type to be reflected upon.
         * @tparam T The target type for reflecting upon.
         * @return The tuple of extracted types.
         */
        template <typename T>
        inline constexpr auto loophole() noexcept
        -> decltype(loophole<T>(std::make_index_sequence<count<T>(0)>()));

        /**
         * Transforms each member type of a tuple into its reference type.
         * @tparam T The tuple's type list.
         */
        template <typename ...T>
        inline constexpr auto reference(tuple_t<T...>) noexcept
        -> tuple_t<T&...>;

        /**
         * Transforms each member type of a tuple into an aligned storage.
         * @tparam T The tuple's type list.
         */
        template <typename ...T>
        inline constexpr auto storage(tuple_t<T...>) noexcept
        -> tuple_t<storage_t<sizeof(T), alignof(T)>...>;
    }

    /**
     * Reflects over an object type and extracts information about its internal
     * property members, transforming it into instantiable tuples.
     * @tparam T The type to be analyzed.
     * @since 1.0
     */
    template <typename T>
    class reflector_t
    {
        static_assert(!std::is_union<T>::value, "union types cannot be reflected");
        static_assert(std::is_trivial<T>::value, "reflected type must be trivial");

        private:
            template <typename A, typename B>
            using is_compatible = std::integral_constant<bool, sizeof(A) == sizeof(B) && alignof(A) == alignof(B)>;

        public:
            using reflection_tuple_t = decltype(detail::loophole<T>());
            using reference_tuple_t = decltype(detail::reference(std::declval<reflection_tuple_t>()));
            using storage_tuple_t = decltype(detail::storage(std::declval<reflection_tuple_t>()));

        static_assert(is_compatible<reflection_tuple_t, T>::value, "reflection tuple is not compatible with type");

        public:
            /**
             * Retrieves the number of members within the reflected type.
             * @return The number of members composing the target type.
             */
            __host__ __device__ inline static constexpr auto count() noexcept -> size_t
            {
                return reflection_tuple_t::count;
            }

            /**
             * Retrieves the offset of a member of the reflected type by its index.
             * @tparam N The index of required member.
             * @return The member offset.
             */
            template <size_t N>
            __host__ __device__ inline static constexpr auto offset() noexcept -> ptrdiff_t
            {
                return offset<N>(storage_tuple_t());
            }

            /**
             * Retrieves the a property member from an instance by its index.
             * @tparam N The requested property member index.
             * @tparam U The compatible introspection target type.
             * @param target The target type instance to retrieve member from.
             * @return The extracted member reference.
             */
            template <size_t N, typename U>
            __host__ __device__ inline static constexpr auto member(U& target) noexcept
            -> typename std::enable_if<
                is_compatible<reflection_tuple_t, U>::value
              , tuple_element_t<reference_tuple_t, N>
            >::type
            {
                using E = tuple_element_t<reflection_tuple_t, N>;
                return *reinterpret_cast<E*>(reinterpret_cast<uint8_t*>(&target) + offset<N>());
            }

          private:
            /**
             * Retrieves the offset of a member in the reflected type by its index.
             * @tparam N The index of the required property member.
             * @param tuple A corresponding reflection's alignment tuple instance.
             * @return The member property's offset.
             */
            template <size_t N>
            __host__ __device__ inline static constexpr auto offset(const storage_tuple_t& tuple) noexcept
            -> ptrdiff_t
            {
                return &tuple.template get<N>().storage[0] - &tuple.template get<0>().storage[0];
            }
    };
}

MUSEQA_END_NAMESPACE

/**
 * Informs the size of a generic reflection tuple, allowing it to be deconstructed.
 * @tparam T The target type for reflection.
 * @since 1.0
 */
template <typename T>
struct std::tuple_size<museqa::utility::reflection_t<T>>
  : std::integral_constant<size_t, museqa::utility::reflection_t<T>::count> {};

/**
 * Retrieves the deconstruction type of a reflection tuple's element.
 * @tparam I The index of the requested tuple element.
 * @tparam T The target type for reflection.
 * @since 1.0
 */
template <size_t I, typename T>
struct std::tuple_element<I, museqa::utility::reflection_t<T>>
  : museqa::identity_t<museqa::utility::tuple_element_t<museqa::utility::reflection_t<T>, I>> {};

MUSEQA_DISABLE_GCC_WARNING_END("-Wnon-template-friend")
MUSEQA_DISABLE_NVCC_WARNING_END(1301)

#endif
