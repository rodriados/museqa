/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Reflection implementation for simple data structures.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Alexandr Poltavsky, Antony Polukhin, Rodrigo Siqueira
 */
#pragma once

#if !defined(MUSEQA_AVOID_REFLECTION)

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
#include <museqa/environment.h>

#if defined(MUSEQA_COMPILER_GCC)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wnon-template-friend"
#endif

/*
 * As the technique used for implementing the loophole that allows us to reflect
 * over the structures is only available from C++14 onwards, we must check whether
 * the current compilation supports it.
 */
#if MUSEQA_CPP >= 201402L

#include <cstddef>
#include <cstdint>
#include <utility>

#include <museqa/utility.hpp>
#include <museqa/utility/tuple.hpp>
#include <museqa/utility/indexer.hpp>

namespace museqa
{
    namespace utility
    {
        /**
         * Reflects over the target data type, that is it extracts information about
         * the member properties of the target type.
         * @tparam T The target data type to be introspected.
         * @since 1.0
         */
        template <typename T>
        class reflector;

        /**
         * Extracts and manages references to each member property of the target
         * type, thus numbering each of the target type's property members and allowing
         * them to be directly accessed or updated.
         * @tparam T The target data type to be introspected.
         * @since 1.0
         */
        template <typename T>
        class reflection : public reflector<T>::reference_tuple
        {
          protected:
            typedef reflector<T> reflector_type;
            typedef typename reflector_type::reference_tuple underlying_tuple;

          public:
            __host__ __device__ inline reflection() noexcept = delete;
            __host__ __device__ inline reflection(const reflection&) noexcept = default;
            __host__ __device__ inline reflection(reflection&&) noexcept = default;

            /**
             * Reflects over an instance and gathers refereces to its members.
             * @param target The target instance to get references from.
             */
            __host__ __device__ inline reflection(T& target) noexcept
              : underlying_tuple {extract(*this, target)}
            {}

            __host__ __device__ inline reflection& operator=(const reflection&) = default;
            __host__ __device__ inline reflection& operator=(reflection&&) = default;

          private:
            /**
             * Retrieves references to the properties of a reflected instance.
             * @tparam U The list of property member types.
             * @tparam I The member types index sequence.
             * @param target The reflected instance to gather references from.
             * @return The new reference tuple instance.
             */
            template <typename ...U, size_t ...I>
            __host__ __device__ inline static auto extract(tuple<indexer<I...>, U...>&, T& target) noexcept
            -> underlying_tuple
            {
                return {reflector_type::template member<I>(target)...};
            }
        };

        namespace impl
        {
            /**
             * Tags a member property type to an index for overload resolution.
             * @tparam T The target type for reflection processing.
             * @tparam N The index of extracted property member type.
             * @since 1.0
             */
            template <typename T, size_t N>
            struct tag
            {
                friend auto latch(tag<T, N>) noexcept;
            };

            /**
             * Injects a friend function to couple a property type to its index.
             * @tparam T The target type for reflection processing.
             * @tparam U The extracted property type.
             * @param N The index of extracted property member type.
             * @since 1.0
             */
            template <typename T, typename U, size_t N>
            struct injector
            {
                /**
                 * Binds the extracted member type to its index within the target
                 * reflected type. This function does not aim to have its concrete
                 * return value used, but only its return type.
                 * @return The extracted type bound to the member index.
                 */
                friend inline auto latch(tag<T, N>) noexcept
                {
                    return typename std::remove_all_extents<U>::type {};
                }
            };

            /**
             * Decoy type responsible for pretending to be a type instance required
             * to build the target reflection type and latching the required type
             * into the injector, so that it can be retrieved later on.
             * @tparam T The target type for reflection processing.
             * @tparam N The index of property member type to extract.
             * @since 1.0
             */
            template <typename T, size_t N>
            struct decoy
            {
                /**
                 * Injects the extracted member type into a latch if it has not
                 * yet been previously done so.
                 * @tparam U The extracted property member type.
                 * @tparam M The index of the property member being processed.
                 */
                template <typename U, size_t M>
                static constexpr auto inject(...) -> injector<T, U, M>;

                /**
                 * Validates whether the type member being processed has already
                 * been reflected over. If yes, avoids latch redeclaration.
                 * @tparam M The index of the property member being processed.
                 */
                template <typename, size_t M>
                static constexpr auto inject(int) -> decltype(latch(tag<T, M>{}));

                /**
                 * Morphs the decoy into the required type for constructing the
                 * target reflection type and injects it into the type latch.
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

            template <typename T, size_t ...I, size_t = sizeof(T {decoy<T, I>{}...})>
            inline constexpr auto count(int) noexcept
            -> size_t { return count<T, I..., sizeof...(I)>(0); }
            /**#@-*/

            /**#@+
             * Extracts the types of the property members within the target reflection
             * object type into an instantiable tuple.
             * @tparam T The target type for reflection processing.
             * @return The tuple of extracted types.
             */
            template <typename T, size_t ...I, typename = decltype(T {decoy<T, I>{}...})>
            inline constexpr auto loophole(indexer<I...>) noexcept
            -> tuple<decltype(latch(tag<T, I>{}))...>;

            template <typename T>
            inline constexpr auto loophole() noexcept
            -> decltype(loophole<T>(typename indexer<count<T>(0)>::type {}));
            /**#@-*/

            /**
             * Transforms each member type of a tuple into its reference type.
             * @tparam T The tuple's type list.
             */
            template <typename ...T>
            inline constexpr auto reference(tuple<T...>) noexcept
            -> tuple<T&...>;

            /**
             * Transforms each member type of a tuple into an aligned storage.
             * @tparam T The tuple's type list.
             */
            template <typename ...T>
            inline constexpr auto storage(tuple<T...>) noexcept
            -> tuple<storage<sizeof(T), alignof(T)>...>;
        }

        /**
         * Reflects over an object type and extracts information about its internal
         * property members, transforming it into instantiable tuples.
         * @tparam T The type to be analyzed.
         * @since 1.0
         */
        template <typename T>
        class reflector
        {
            static_assert(!std::is_union<T>::value, "union types cannot be reflected");
            static_assert(std::is_trivial<T>::value, "reflected type must be trivial");
            static_assert(std::is_class<T>::value, "reflected type must be class or struct");

          public:
            using reflection_tuple = decltype(impl::loophole<T>());
            using reference_tuple = decltype(impl::reference(std::declval<reflection_tuple>()));
            using storage_tuple = decltype(impl::storage(std::declval<reflection_tuple>()));

            static_assert(sizeof(reflection_tuple) == sizeof(T), "reflection tuple is not compatible with type");
            static_assert(alignof(reflection_tuple) == alignof(T), "reflection tuple is not compatible with type");

          public:
            /**
             * Retrieves the number of members within the reflected type.
             * @return The number of members composing the target type.
             */
            __host__ __device__ inline static constexpr auto count() noexcept -> size_t
            {
                return reflection_tuple::count;
            }

            /**
             * Retrieves the offset of a member of the reflected type by its index.
             * @tparam N The index of required member.
             * @return The member offset.
             */
            template <size_t N>
            __host__ __device__ inline static constexpr auto offset() noexcept -> ptrdiff_t
            {
                return offset<N>(storage_tuple {});
            }

            /**
             * Retrieves the a property member from an instance by its index.
             * @tparam N The requested property member index.
             * @param target The target type instance to retrieve member from.
             * @return The extracted member reference.
             */
            template <size_t N>
            __host__ __device__ inline static constexpr auto member(T& target) noexcept
            -> tuple_element<reference_tuple, N>
            {
                using U = tuple_element<reflection_tuple, N>;
                return *reinterpret_cast<U*>(reinterpret_cast<uint8_t*>(&target) + offset<N>());
            }

          private:
            /**
             * Retrieves the offset of a member in the reflected type by its index.
             * @tparam N The index of the required property member.
             * @param tuple A corresponding reflection's alignment tuple instance.
             * @return The member property's offset.
             */
            template <size_t N>
            __host__ __device__ inline static constexpr auto offset(const storage_tuple& tuple) noexcept
            -> ptrdiff_t
            {
                return &tuple.template get<N>().storage[0] - &tuple.template get<0>().storage[0];
            }
        };
    }
}

#endif

#if defined(MUSEQA_COMPILER_GCC)
  #pragma GCC diagnostic pop
#endif

#endif
