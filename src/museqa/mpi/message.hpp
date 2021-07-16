/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A general type-agnostic MPI message wrapper implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#if !defined(MUSEQA_AVOID_MPI)

#include <cstdint>

#include <museqa/mpi/type.hpp>
#include <museqa/memory/pointer/shared.hpp>

namespace museqa
{
    namespace mpi
    {
        /**
         * Represents a type-agnostic message, which can hold a reference to a MPI
         * message of any type.
         * @since 1.0
         */
        struct message
        {
            typedef memory::pointer::shared<void> pointer_type;

            pointer_type ptr;       /// The message's contents pointer.
            mpi::type::id type;     /// The message's contents type description.
            int32_t size = 1;       /// The total number of elements in message.

            inline message() noexcept = default;
            inline message(const message&) noexcept = default;
            inline message(message&&) noexcept = default;

            /**
             * Initializes a new message from a generic pointer.
             * @tparam T The message's contents type.
             * @param ptr The message's contents pointer.
             * @param size The total number of elements in message.
             */
            template <typename T>
            inline message(memory::pointer::shared<T>& ptr, size_t size = 1) noexcept
              : ptr {ptr}
              , type {mpi::type::identify<T>()}
              , size {static_cast<int32_t>(size)}
            {}

            /**
             * Initializes a new message from an untyped pointer.
             * @param ptr The message's contents pointer.
             * @param type The message's contents type identifier.
             * @param size The total number of elements in message.
             */
            inline message(pointer_type& ptr, mpi::type::id type, size_t size = 1) noexcept
              : ptr {ptr}
              , type {type}
              , size {static_cast<int32_t>(size)}
            {}

            inline message& operator=(const message&) noexcept = default;
            inline message& operator=(message&&) noexcept = default;
        };
    }

    namespace factory
    {
        namespace mpi
        {
            /**
             * Builds a new empty MPI message instance whose allocated memory is
             * enough to store the given number of elements.
             * @tparam T The message's contents type.
             * @param size The total number of elements in message.
             * @return The new empty message instance.
             */
            template <typename T>
            inline auto message(size_t size = 1) noexcept -> museqa::mpi::message
            {
                auto ptr = memory::pointer::shared<void> {operator new(size * sizeof(T))};
                auto type = museqa::mpi::type::identify<T>();
                return {ptr, type, size};
            }
        }
    }
}

#endif
