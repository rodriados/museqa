/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A MPI type descriptor implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

#if !defined(MUSEQA_AVOID_MPI)

#include <mpi.h>

#include <array>
#include <cstdint>
#include <utility>

#include <museqa/utility.hpp>
#include <museqa/utility/tuple.hpp>
#include <museqa/mpi/common.hpp>

#if !defined(MUSEQA_AVOID_REFLECTION)
  #include <museqa/utility/reflection.hpp>
#endif

namespace museqa
{
    namespace mpi
    {
        namespace type
        {
            /**
             * The type for a datatype identifier instance. An instance of a datatype
             * descriptor must exist for all types that may trasit via MPI.
             * @since 1.0
             */
            using id = MPI_Datatype;

            /**
             * Creates the description for a type that may transit as a message
             * via MPI. A descriptor must receive the type's member properties pointers
             * as constructor parameters in order to describe a type.
             * @see museqa::mpi::type::describe
             * @since 1.0 
             */
            struct descriptor
            {
                type::id id;                    /// The target type's raw identifier.

                inline descriptor() noexcept = delete;
                inline descriptor(const descriptor&) noexcept = delete;
                inline descriptor(descriptor&&) noexcept = default;

                template <typename T, typename ...U>
                inline descriptor(U T::*...);

                template <size_t ...I, typename ...T>
                inline descriptor(const utility::tuple<utility::indexer<I...>, T...>&);

                inline ~descriptor();

                inline descriptor& operator=(const descriptor&) noexcept = delete;
                inline descriptor& operator=(descriptor&&) noexcept = delete;
            };

            /**
             * Describes a type and allows it to be sent to different nodes via MPI.
             * @tparam T The type to be described.
             * @return The target type's descriptor instance.
             * @see museqa::mpi::type::descriptor
             */
            template <typename T>
            inline descriptor describe();

          #if MUSEQA_CPP >= 201402L && !defined(MUSEQA_AVOID_REFLECTION)
            /**
             * Creates a type description for MPI transportation using reflection
             * over the target type.
             * @tparam T The type to be described.
             * @return The target type's description instance.
             */
            template <typename T>
            inline descriptor describe()
            {
                return descriptor {typename utility::reflector<T>::reflection_tuple {}};
            }
          #endif

            /**
             * Identifies the given type by retrieving its raw datatype id.
             * @tparam T The type to get the raw datatype id of.
             * @return The requested type's id.
             */
            template <typename T>
            inline type::id identify()
            {
                static_assert(!std::is_union<T>::value, "union types cannot be sent via MPI");
                static_assert(!std::is_reference<T>::value, "references cannot be sent via MPI");
                static_assert(std::is_trivial<T>::value, "only trivial types can be sent via MPI");

                static auto descriptor = describe<T>();
                return descriptor.id;
            }

            /**#@+
             * Specializations for identifiers of built-in types. These native types
             * have their identities created built-in by MPI and can be used directly.
             * @since 1.0
             */
            template <> inline type::id identify<bool>()     { return MPI_C_BOOL; };
            template <> inline type::id identify<char>()     { return MPI_CHAR; };
            template <> inline type::id identify<float>()    { return MPI_FLOAT; };
            template <> inline type::id identify<double>()   { return MPI_DOUBLE; };
            template <> inline type::id identify<int8_t>()   { return MPI_INT8_T; };
            template <> inline type::id identify<int16_t>()  { return MPI_INT16_T; };
            template <> inline type::id identify<int32_t>()  { return MPI_INT32_T; };
            template <> inline type::id identify<int64_t>()  { return MPI_INT64_T; };
            template <> inline type::id identify<uint8_t>()  { return MPI_UINT8_T; };
            template <> inline type::id identify<uint16_t>() { return MPI_UINT16_T; };
            template <> inline type::id identify<uint32_t>() { return MPI_UINT32_T; };
            template <> inline type::id identify<uint64_t>() { return MPI_UINT64_T; };
            /**#@-*/
        }

        namespace impl
        {
            /**
             * Creates a description identifier for a custom type.
             * @tparam T The list of member types within the custom type.
             * @param array The list of the type's property member offsets.
             * @return The target type's description identifier.
             */
            template <typename ...T>
            inline static auto describe(const std::array<ptrdiff_t, sizeof...(T)>& array) -> type::id
            {
                type::id result;
                constexpr const size_t count = sizeof...(T);

                int blocks[count] = {utility::max((int) std::extent<T>::value, 1)...};
                type::id types[count] = {type::identify<typename std::remove_extent<T>::type>()...};
                const MPI_Aint *offsets = array.data();

                mpi::check(MPI_Type_create_struct(count, blocks, offsets, types, &result));
                mpi::check(MPI_Type_commit(&result));

                return result;
            }
        }

        /**
         * Constructs a new type description. A type description is needed whenever
         * one needs to send a type instance to different MPI nodes.
         * @tparam T The type to be described through its property members.
         * @tparam U The target type member types list.
         * @param members The target type property member pointers.
         */
        template <typename T, typename ...U>
        inline type::descriptor::descriptor(U T::*... members)
          : id {impl::describe<U...>({(char*) &(((T*) nullptr)->*members) - (char*) nullptr...})}
        {}

        /**
         * Constructs a new type description from a tuple. The given tuple must contain
         * aligned members to those of the original type.
         * @tparam T The list of property member types within the tuple.
         * @param tuple A type-describing tuple instance.
         */
        template <size_t ...I, typename ...T>
        inline type::descriptor::descriptor(const utility::tuple<utility::indexer<I...>, T...>& tuple)
          : id {impl::describe<T...>({(char*) &tuple.template get<I>() - (char*) &tuple.template get<0>()...})}
        {}

        /**
         * Frees up the resources needed for storing a type's description. Effectively,
         * after the execution of this destructor, the type description identifier
         * wrapped by the object is in an invalid state and should not be used.
         * @see museqa::mpi::type::descriptor
         */
        inline type::descriptor::~descriptor()
        {
            mpi::check(MPI_Type_free(&id));
        }
    }
}

#endif
