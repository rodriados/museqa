/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A MPI type descriptor implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

#if !defined(MUSEQA_AVOID_MPI)

#include <mpi.h>

#include <cstdint>
#include <utility>

#include <museqa/utility.hpp>
#include <museqa/mpi/common.hpp>

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
             * @see museqa::mpi::describer
             * @since 1.0 
             */
            struct descriptor
            {
                typedef type::id identifier_type;       /// The type identifier's type.
                identifier_type id;                     /// The target type's identifier.

                inline descriptor() noexcept = delete;
                inline descriptor(const descriptor&) noexcept = delete;
                inline descriptor(descriptor&&) noexcept = default;

                template <typename T, typename ...U>
                inline descriptor(U T::*...);

                inline ~descriptor();

                inline descriptor& operator=(const descriptor&) noexcept = delete;
                inline descriptor& operator=(descriptor&&) noexcept = delete;
            };
        }

        /**
         * Describes a type and allows it to be sent to different nodes via MPI.
         * @tparam T The type to be described.
         * @return The target type's descriptor instance.
         * @see museqa::mpi::descriptor
         */
        template <typename T>
        inline type::descriptor describe()
        {
            return {};
        }

        namespace type
        {
            /**
             * Retrieves a datatype identifier for the given type. To perform this
             * task, a descriptor must be available for the requested type.
             * @tparam T The type to be identified.
             * @since 1.0
             */
            template <typename T>
            struct identifier
            {
                static_assert(!std::is_union<T>::value, "union types cannot be sent via MPI");
                static_assert(!std::is_reference<T>::value, "references cannot be sent via MPI");
                static_assert(std::is_trivial<T>::value, "only trivial types can be sent via MPI");

                /**
                 * Requests the target type's identifier. This method invokes the
                 * type describer only once and caches its result for later use.
                 * @return The type's identifier.
                 */
                inline static auto get() -> id
                {
                    static auto descriptor = describe<T>();
                    return descriptor.id;
                }
            };

            /**
             * Identifies the given type by retrieving its raw datatype id.
             * @tparam T The type to get the raw datatype id of.
             * @return The requested type's id.
             */
            template <typename T> inline id identify() { return identifier<T>::get(); }

            /**#@+
             * Specializations for identifiers of built-in types. These native types
             * have their identities created built-in by MPI and can be used directly.
             * @since 1.0
             */
            template <> inline id identify<bool>()     { return MPI_C_BOOL; };
            template <> inline id identify<char>()     { return MPI_CHAR; };
            template <> inline id identify<float>()    { return MPI_FLOAT; };
            template <> inline id identify<double>()   { return MPI_DOUBLE; };
            template <> inline id identify<int8_t>()   { return MPI_INT8_T; };
            template <> inline id identify<int16_t>()  { return MPI_INT16_T; };
            template <> inline id identify<int32_t>()  { return MPI_INT32_T; };
            template <> inline id identify<int64_t>()  { return MPI_INT64_T; };
            template <> inline id identify<uint8_t>()  { return MPI_UINT8_T; };
            template <> inline id identify<uint16_t>() { return MPI_UINT16_T; };
            template <> inline id identify<uint32_t>() { return MPI_UINT32_T; };
            template <> inline id identify<uint64_t>() { return MPI_UINT64_T; };
            /**#@-*/
        }

        namespace impl
        {
            /**
             * Creates a description identifier for a custom type.
             * @tparam T The custom type to be described.
             * @tparam U The target type's property member types.
             * @param members The static pointers for the target type's properties.
             * @return The target type's description identifier.
             */
            template <typename T, typename ...U>
            inline static auto describe(U T::*... members) -> type::id
            {
                constexpr const size_t count = sizeof...(U);

                type::id result;

                int blocks[count] = {utility::max((int) std::extent<U>::value, 1)...};
                MPI_Aint offsets[count] = {((char *) &(((T*) nullptr)->*members) - (char *) nullptr)...};
                type::id types[count] = {type::identify<typename std::remove_extent<U>::type>()...};

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
          : id {impl::describe(members...)}
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
