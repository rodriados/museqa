/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file MPI communicators wrapper and helper functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#if !defined(MUSEQA_AVOID_MPI)

#include <mpi.h>

#include <cstdint>
#include <utility>

#include <museqa/node.hpp>
#include <museqa/utility.hpp>
#include <museqa/mpi/common.hpp>
#include <museqa/memory/pointer/weak.hpp>
#include <museqa/memory/pointer/shared.hpp>

namespace museqa
{
    namespace mpi
    {
        /**
         * The type of a raw MPI communicator channel identifier. An identifier for
         * a channel must exist in order to use MPI's collective operations.
         * @since 1.0
         */
        using channel = MPI_Comm;

        /**
         * Represents a communicator channel, allowing messages to be exchanged between
         * different nodes within a MPI execution.
         * @since 1.0
         */
        class communicator : private memory::pointer::shared<void>
        {
            friend void mpi::init(int&, char**&);
            friend void mpi::finalize();

          private:
            typedef mpi::channel reference_type;
            typedef memory::pointer::shared<void> underlying_type;

          public:
            const mpi::node rank = 0;           /// The current node's rank within communicator.
            const int32_t size = 0;             /// The total number of nodes in communicator.

          public:
            inline communicator() noexcept = default;
            inline communicator(const communicator&) noexcept = default;
            inline communicator(communicator&&) noexcept = default;

            /**
             * Initializes a new communicator from a raw channel reference.
             * @param channel The channel to build a new communicator from.
             */
            inline communicator(reference_type channel) noexcept(museqa::unsafe)
              : communicator {build(channel)}
            {}

            inline communicator& operator=(const communicator&) = default;
            inline communicator& operator=(communicator&&) = delete;

            /**
             * Converts the communicator into the raw MPI communicator reference,
             * allowing it to be seamlessly used with native MPI functions.
             * @return The underlying communicator channel.
             */
            inline operator reference_type() const noexcept
            {
                return (reference_type) this->m_ptr;
            }

            communicator split(int, int = mpi::any) noexcept(museqa::unsafe);
            communicator duplicate() noexcept(museqa::unsafe);

          protected:
            /**
             * Creates a new communicator instance from a raw channel reference.
             * @param rank The current node's rank within the communicator.
             * @param size The total number of nodes connected to the channel.
             * @param channel The channel to build a new communicator from.
             */
            inline communicator(mpi::node rank, int32_t size, reference_type channel) noexcept
              : communicator {rank, size, wrap(channel)}
            {}

            /**
             * Creates a new communicator instance from a wrapped channel.
             * @param rank The current node's rank within the communicator.
             * @param size The total number of nodes connected to the channel.
             * @param channel The wrapped channel instance.
             */
            inline communicator(mpi::node rank, int32_t size, const underlying_type& channel) noexcept
              : underlying_type {channel}
              , rank {rank}
              , size {size}
            {}

            static auto build(reference_type) noexcept(museqa::unsafe) -> communicator;

          private:
            /**
             * Helper method for initializing a global communicator.
             * @param channel The raw MPI channel reference.
             * @return The global communicator instance.
             */
            inline static auto global(reference_type channel) noexcept -> communicator
            {
                auto ptr = memory::pointer::weak<void> {channel};
                return {node::master, 1, ptr};
            }

            /**
             * Wraps a raw channel reference into a pointer with automatic duration.
             * @param channel The raw MPI channel reference to be wrapped.
             * @return The wrapped pointer instance.
             */
            inline static auto wrap(reference_type channel) noexcept -> underlying_type
            {
                auto destructor = [](void *ptr) { mpi::check(MPI_Comm_free((reference_type*) &ptr)); };
                return underlying_type {channel, destructor};
            }
        };

        /**
         * The default globally available communicator for MPI collective messages.
         * This communicator allows us to readily transfer messages between nodes
         * without the need to previously allocate a communicator.
         * @since 1.0
         */
        extern communicator world;
    }
}

#endif
