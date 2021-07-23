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

#include <museqa/node.hpp>
#include <museqa/utility.hpp>
#include <museqa/mpi/common.hpp>
#include <museqa/memory/buffer.hpp>
#include <museqa/memory/pointer/weak.hpp>
#include <museqa/memory/pointer/shared.hpp>

namespace museqa
{
    namespace mpi
    {
        /**
         * The type of a raw MPI communicator channel identifier. An identifier
         * for a channel must exist in order to use MPI's collective operations.
         * @since 1.0
         */
        using channel = MPI_Comm;

        /**
         * Represents a communicator channel, allowing messages to be exchanged
         * between different nodes within a MPI execution.
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
            inline communicator(reference_type channel)
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

            communicator split(int, int = mpi::any) const;
            communicator duplicate() const;

            static communicator create(const communicator&, const class group&, mpi::tag = mpi::any);

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

            static communicator build(reference_type);

          private:
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
         * Represents a group of nodes within a communicator channel. Operations
         * over node groups are local, therefore no communication between nodes take
         * place before converting the group back to a channel.
         * @since 1.0
         */
        class group : private memory::pointer::shared<void>
        {
          private:
            typedef MPI_Group reference_type;
            typedef memory::pointer::shared<void> underlying_type;

          public:
            inline group() noexcept = default;
            inline group(const group&) noexcept = default;
            inline group(group&&) noexcept = default;

            /**
             * Initializes a node group from a communicator instance.
             * @param comm The communicator to retrieve the group from.
             */
            inline group(const communicator& comm) noexcept(!safe)
            {
                mpi::check(MPI_Comm_group(comm, (reference_type*) &this->m_ptr));
            }

            inline group& operator=(const group&) = default;
            inline group& operator=(group&&) = default;

            /**
             * Converts the group into a raw MPI node group reference, allowing it
             * to be seamlessly used with native MPI functions.
             * @return The underlying node group reference.
             */
            inline operator reference_type() const noexcept
            {
                return (reference_type) this->m_ptr;
            }

            group operator+(const group&) const noexcept(!safe);
            group operator-(const group&) const noexcept(!safe);

            /**
             * Creates a new node group by without some nodes from an existing group.
             * @param group The group instance to have nodes excluded in new group.
             * @param nodes The list of excluded nodes in new group.
             * @return The new node group instance.
             */
            inline static group exclude(const group& group, std::initializer_list<mpi::node> nodes)
            {
                return exclude(group, nodes.begin(), nodes.size());
            }

            /**
             * Creates a new node group by without some nodes from an existing group.
             * @param group The group instance to have nodes excluded in new group.
             * @param nodes The buffer of excluded nodes in new group.
             * @return The new node group instance.
             */
            inline static group exclude(const group& group, const memory::buffer<mpi::node>& nodes)
            {
                return exclude(group, nodes.begin(), nodes.capacity());
            }

            /**
             * Creates a new node group by selecting nodes to form a new group.
             * @param group The base group instance to get nodes for new group.
             * @param nodes The list of nodes to be included in new group.
             * @return The new node group instance.
             */
            inline static group include(const group& group, std::initializer_list<mpi::node> nodes)
            {
                return include(group, nodes.begin(), nodes.size());
            }

            /**
             * Creates a new node group by selecting nodes to form a new group.
             * @param group The base group instance to get nodes for new group.
             * @param nodes The buffer of nodes to be included in new group.
             * @return The new node group instance.
             */
            inline static group include(const group& group, const memory::buffer<mpi::node>& nodes)
            {
                return include(group, nodes.begin(), nodes.capacity());
            }

            static group exclude(const group&, const mpi::node*, size_t) noexcept(!safe);
            static group include(const group&, const mpi::node*, size_t) noexcept(!safe);

            static group intersection(const group&, const group&) noexcept(!safe);

          protected:
            /**
             * Creates a new node group instance from a raw group reference.
             * @param ref The group reference to build a new instance from.
             */
            inline group(reference_type ref) noexcept
              : group {wrap(ref)}
            {}

            /**
             * Creates a new group instance from a wrapped node group reference.
             * @param ref The wrapped node group reference.
             */
            inline group(const underlying_type& ref) noexcept
              : underlying_type {ref}
            {}

          private:
            /**
             * Wraps a raw group reference into a pointer with automatic duration.
             * @param ref The raw MPI group reference to be wrapped.
             * @return The wrapped pointer instance.
             */
            inline static auto wrap(reference_type ref) noexcept -> underlying_type
            {
                auto destructor = [](void *ptr) { mpi::check(MPI_Group_free((reference_type*) &ptr)); };
                return underlying_type {ref, destructor};
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
