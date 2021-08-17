/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Collective operations between nodes implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#if !defined(MUSEQA_AVOID_MPI)

#include <mpi.h>

#include <numeric>

#include <museqa/node.hpp>
#include <museqa/utility.hpp>
#include <museqa/utility/tuple.hpp>

#include <museqa/mpi/type.hpp>
#include <museqa/mpi/common.hpp>
#include <museqa/mpi/lambda.hpp>
#include <museqa/mpi/status.hpp>
#include <museqa/mpi/payload.hpp>
#include <museqa/mpi/message.hpp>
#include <museqa/mpi/communicator.hpp>

namespace museqa
{
    namespace mpi
    {
        /**
         * Synchronizes the execution of all nodes within a channel.
         * @param comm The communicator channel the operation should apply to.
         */
        inline void barrier(const communicator& comm = world) noexcept(!safe)
        {
            mpi::check(MPI_Barrier(comm));
        }

        /**
         * Broadcasts a message to all nodes in given communicator channel.
         * @param in The message to be broadcast to all nodes.
         * @param root The operation's root node.
         * @param comm The communicator channel this operation applies to.
         * @return The message that has been received.
         */
        inline message mbroadcast(
            const message& in
          , node root = museqa::node::master
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            auto out = (root == comm.rank) ? in : factory::mpi::message(in.type, in.size);
            mpi::check(MPI_Bcast(out.ptr, out.size, out.type, root, comm));
            return out;
        }

        /**
         * Broadcasts a generic message to all nodes in the given communicator.
         * @tparam T The type of message data to broadcast.
         * @param data The message to be broadcast to all nodes.
         * @param size The number of elements to be broadcast.
         * @param root The operation's root node.
         * @param comm The communicator channel this operation applies to.
         * @return The operation's resulting message.
         */
        template <typename T>
        inline typename payload<T>::return_type broadcast(
            T *data
          , size_t size
          , node root = museqa::node::master
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            payload<T> msg {data, size};
            return mpi::mbroadcast(msg, root, comm);
        }

        /**
         * Broadcasts a container to all nodes in the given communicator.
         * @tparam T The type of the container's data to broadcast.
         * @param data The container to be broadcast to all nodes.
         * @param root The operation's root node.
         * @param comm The communicator channel this operation applies to.
         * @return The broadcast message.
         */
        template <typename T>
        inline typename payload<T>::return_type broadcast(
            T& data
          , node root = museqa::node::master
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            message msg = payload<T> {data};
            msg.size = mpi::broadcast(&msg.size, 1, root, comm);
            return mpi::mbroadcast(msg, root, comm);
        }

        /**
         * Inspects an incoming message and retrieves its status.
         * @param src The message's source node.
         * @param tag The message's identifying tag.
         * @param comm The communicator channel the message was sent through.
         * @return The inspected message status.
         */
        inline status probe(
            node src = any
          , mpi::tag tag = any
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            status::raw_type stt;
            mpi::check(MPI_Probe(src, tag, comm, &stt));
            return status {stt};
        }

        /**
         * Sends a message to a node connected to the communicator channel.
         * @param in The message to be sent to the destination node.
         * @param dest The node within the channel to send message to.
         * @param tag The message's identifying tag.
         * @param comm The communicator channel to send a message through.
         */
        inline void msend(
            const message& in
          , node dest = museqa::node::master
          , mpi::tag tag = any
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            if (tag < 0) { tag = MPI_TAG_UB; }
            mpi::check(MPI_Send(in.ptr, in.size, in.type, dest, tag, comm));
        }

        /**
         * Sends a message to a node connected to the communicator channel.
         * @tparam T The type of message data to send.
         * @param data The message to send to the specified node.
         * @param size The number of elements to be sent.
         * @param dest The message's destination node.
         * @param tag The message's identifying tag.
         * @param comm The communicator channel to send a message through.
         */
        template <typename T>
        inline void send(
            T *data
          , size_t size
          , node dest = museqa::node::master
          , mpi::tag tag = any
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            payload<T> msg {data, size};
            mpi::msend(msg, dest, tag, comm);
        }

        /**
         * Sends a container to a node connected to the communicator channel.
         * @tparam T The type of the container's data to send.
         * @param data The container to send to the specified node.
         * @param dest The message's destination node.
         * @param tag The message's identifying tag.
         * @param comm The communicator channel to send a message through.
         */
        template <typename T>
        inline void send(
            T& data
          , node dest = museqa::node::master
          , mpi::tag tag = any
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            payload<T> msg {data};
            mpi::msend(msg, dest, tag, comm);
        }

        /**
         * Receives and waits for an incoming message from source node.
         * @param out The message to be received from the source node.
         * @param src The node within the channel to receive the message from.
         * @param tag The message's identifying tag.
         * @param comm The communicator to receive the receive through.
         * @return The received message and its status.
         */
        inline message mreceive(
            message& out
          , node src = any
          , mpi::tag tag = any
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            status::raw_type stt;
            mpi::check(MPI_Recv(out.ptr, out.size, out.type, src, tag, comm, &stt));
            last_status = status {stt};
            return out;
        }

        /**
         * Receives a message from a node connected to the communicator channel.
         * @tparam T The type of message data to receive.
         * @param src The message's source node.
         * @param tag The message's identifying tag.
         * @param comm The communicator channel to receive a message through.
         * @return The received message.
         */
        template <typename T>
        inline typename payload<T>::return_type receive(
            node src = any
          , mpi::tag tag = any
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            auto stt = mpi::probe(src, tag, comm);
            using E = typename payload<T>::element_type;
            auto msg = factory::mpi::message<E>(stt.count<E>());
            return mpi::mreceive(msg, src, tag, comm);
        }

        /**
         * Reduces a message from all nodes to a single one in all nodes.
         * @param in The current process message to be reduced.
         * @param lambda The reduce operator reference.
         * @param comm The communicator channel to reduce messages with.
         * @return The reduced message.
         */
        inline message mallreduce(
            const message& in
          , lambda::id lambda
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            auto out = factory::mpi::message(in.type, in.size);
            mpi::check(MPI_Allreduce(in.ptr, out.ptr, out.size, out.type, impl::lambda::active = lambda, comm));
            return out;
        }

        /**
         * Reduces a message from all nodes and sends result to all nodes.
         * @tparam T The type of message data to be reduced.
         * @param data The message data to be reduced.
         * @param size The number of elements to be reduced.
         * @param lambda The reduce operator reference.
         * @param comm The communicator channel to reduce messages with.
         * @return The reduced message.
         */
        template <typename T>
        inline typename payload<T>::return_type allreduce(
            T *data
          , size_t size
          , lambda::id lambda
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            payload<T> msg {data, size};
            return mpi::mallreduce(msg, lambda, comm);
        }

        /**
         * Reduces a container from all nodes and sends result to all nodes.
         * @tparam T The type of container data to be reduced.
         * @param data The container to be reduced.
         * @param lambda The reduce operator reference.
         * @param comm The communicator channel to reduce messages with.
         * @return The reduced message.
         */
        template <typename T>
        inline typename payload<T>::return_type allreduce(
            T& data
          , lambda::id lambda
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            payload<T> msg {data};
            return mpi::mallreduce(msg, lambda, comm);
        }

        /**
         * Reduces a message from all nodes to a single one in the root node.
         * @param in The current process' message to be reduced.
         * @param lambda The reduce operator reference.
         * @param root The reduce operation's root node.
         * @param comm The communicator channel to reduce messages with.
         * @return The reduced message.
         */
        inline message mreduce(
            const message& in
          , lambda::id lambda
          , node root = museqa::node::master
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            auto out = (root != comm.rank) ? in : factory::mpi::message(in.type, in.size);
            mpi::check(MPI_Reduce(in.ptr, out.ptr, out.size, out.type, impl::lambda::active = lambda, root, comm));
            return out;
        }

        /**
         * Reduces a message from all nodes into the root node.
         * @tparam T The type of message data to be reduced.
         * @param data The message data to be reduced.
         * @param size The number of elements to be reduced.
         * @param lambda The reduce operator reference.
         * @param root The reduce operation's root node.
         * @param comm The communicator channel to reduce messages with.
         * @return The reduced message.
         */
        template <typename T>
        inline typename payload<T>::return_type reduce(
            T *data
          , size_t size
          , lambda::id lambda
          , node root = museqa::node::master
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            payload<T> msg {data, size};
            return mpi::mreduce(msg, lambda, root, comm);
        }

        /**
         * Reduces a container from all nodes into the root node.
         * @tparam T The type of container data to be reduced.
         * @param data The container to be reduced.
         * @param lambda The reduce operator reference.
         * @param root The reduce operation's root node.
         * @param comm The communicator channel to reduce messages with.
         * @return The reduced message.
         */
        template <typename T>
        inline typename payload<T>::return_type reduce(
            T& data
          , lambda::id lambda
          , node root = museqa::node::master
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            payload<T> msg {data};
            return mpi::mreduce(msg, lambda, root, comm);
        }

        /**
         * Gathers messages from all nodes with channel and delivers to all nodes.
         * @param in The current process' message to be gathered.
         * @param comm The communicator channel to gather messages from.
         * @return The gathered message.
         */
        inline message mallgather(
            const message& in
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            auto out = factory::mpi::message(in.type, in.size * comm.size);
            mpi::check(MPI_Allgather(in.ptr, in.size, out.type, out.ptr, in.size, out.type, comm));
            return out;
        }

        /**
         * Gathers messages of different sizes from all nodes to all nodes.
         * @param in The current process' message to be gathered.
         * @param size The list of message sizes by each node.
         * @param disp The displacement of each node's messages.
         * @param comm The communicator channel to gather messages from.
         * @return The gathered message.
         */
        inline message mallgatherv(
            const message& in
          , const message& size
          , const message& disp
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            const int *sptr = (const int *) size.ptr;
            const int *dptr = (const int *) disp.ptr;
            auto count = std::accumulate(sptr, sptr + comm.size, 0);
            auto out = factory::mpi::message(in.type, count);
            mpi::check(MPI_Allgatherv(in.ptr, in.size, out.type, out.ptr, sptr, dptr, out.type, comm));
            return out;
        }

        /**
         * Gathers data from all nodes in all nodes within the given channel.
         * @tparam T The type of message data to be gathered.
         * @param data The message data to be gathered.
         * @param size The number of message elements of each node.
         * @param comm The communicator channel to gather messages with.
         * @return The gathered message.
         */
        template <typename T>
        inline typename payload<T>::return_type allgather(
            T *data
          , size_t size
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            payload<T> msg {data, size};
            return mpi::mallgather(msg, comm);
        }

        /**
         * Gathers containers from all nodes in all nodes within the given channel.
         * @tparam T The type of message data to be gathered.
         * @param data The message data to be gathered.
         * @param comm The communicator channel to gather messages with.
         * @return The gathered message.
         */
        template <typename T>
        inline typename payload<T>::return_type allgather(
            T& data
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            message msg = payload<T> {data};
            payload<int> size = mpi::allgather(&msg.size, 1, comm);
            payload<int> disp = factory::buffer<int>(comm.size);

            bool homogeneous = true;

            for (size_t i = 0; i < comm.size; ++i) {
                homogeneous = homogeneous && (size[i] == size[0]);
                disp[i] = (i <= 0) ? 0 : (disp[i - 1] + size[i - 1]);
            }

            return homogeneous
                ? mpi::mallgather(msg, comm)
                : mpi::mallgatherv(msg, size, disp, comm);
        }

        /**
         * Gathers messages from all nodes into the root node.
         * @param in The current process' message to be gathered.
         * @param root The target node for gathering the messages.
         * @param comm The communicator channel to gather messages from.
         * @return The gathered message.
         */
        inline message mgather(
            const message& in
          , node root = museqa::node::master
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            auto out = (root != comm.rank) ? in : factory::mpi::message(in.type, in.size * comm.size);
            mpi::check(MPI_Gather(in.ptr, in.size, out.type, out.ptr, in.size, out.type, root, comm));
            return out;
        }

        /**
         * Gathers messages of different size from all nodes into the root node.
         * @param in The current process' message to be gathered.
         * @param size The list of message sizes by each node.
         * @param disp The displacement of each node's messages.
         * @param root The target node for gathering the messages.
         * @param comm The communicator channel to gather messages from.
         * @return The gathered message.
         */
        inline message mgatherv(
            const message& in
          , const message& size
          , const message& disp
          , node root = museqa::node::master
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            const int *sptr = (const int *) size.ptr;
            const int *dptr = (const int *) disp.ptr;
            auto count = std::accumulate(sptr, sptr + comm.size, 0);
            auto out = (root != comm.rank) ? in : factory::mpi::message(in.type, count);
            mpi::check(MPI_Gatherv(in.ptr, in.size, out.type, out.ptr, sptr, dptr, out.type, root, comm));
            return out;
        }

        /**
         * Gathers data from all nodes into the root node within the given channel.
         * @tparam T The type of message data to be gathered.
         * @param data The message data to be gathered.
         * @param size The number of message elements of each node.
         * @param root The target node for gathering the messages.
         * @param comm The communicator channel to gather messages with.
         * @return The gathered message.
         */
        template <typename T>
        inline typename payload<T>::return_type gather(
            T *data
          , size_t size
          , node root = museqa::node::master
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            payload<T> msg {data, size};
            return mpi::mgather(msg, root, comm);
        }

        /**
         * Gathers containers from all nodes into the root node within the channel.
         * @tparam T The type of container data to be gathered.
         * @param data The container data to be gathered.
         * @param root The target node for gathering the messages.
         * @param comm The communicator channel to gather messages with.
         * @return The gathered message.
         */
        template <typename T>
        inline typename payload<T>::return_type gather(
            T& data
          , node root = museqa::node::master
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            message msg = payload<T> {data};
            payload<int> size = mpi::allgather(&msg.size, 1, comm);
            payload<int> disp = factory::buffer<int>(comm.size);

            bool homogeneous = true;

            for (size_t i = 0; i < comm.size; ++i) {
                homogeneous = homogeneous && (size[i] == size[0]);
                disp[i] = (i <= 0) ? 0 : (disp[i - 1] + size[i - 1]);
            }

            return homogeneous
                ? mpi::mgather(msg, root, comm)
                : mpi::mgatherv(msg, size, disp, root, comm);
        }

        /**
         * Scatters a message from root node to all nodes in channel communicator.
         * @param in The root process' message to be scattered.
         * @param root The root node for scattering the messages from.
         * @param comm The communicator channel to scatter messages with.
         * @return The scattered message.
         */
        inline message mscatter(
            const message& in
          , node root = museqa::node::master
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            auto out = factory::mpi::message(in.type, in.size / comm.size);
            mpi::check(MPI_Scatter(in.ptr, out.size, out.type, out.ptr, out.size, out.type, root, comm));
            return out;
        }

        /**
         * Scatters a message from root node to all nodes in channel communicator.
         * @param in The root process' message to be scattered.
         * @param size The list of message sizes for each node.
         * @param disp The displacement of each node's messages.
         * @param root The root node for scattering the messages from.
         * @param comm The communicator channel to scatter messages with.
         * @return The scattered message.
         */
        inline message mscatterv(
            const message& in
          , const message& size
          , const message& disp
          , node root = museqa::node::master
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            const int *sptr = (const int *) size.ptr;
            const int *dptr = (const int *) disp.ptr;
            auto out = factory::mpi::message(in.type, sptr[comm.rank]);
            mpi::check(MPI_Scatterv(in.ptr, sptr, dptr, out.type, out.ptr, out.size, out.type, root, comm));
            return out;
        }

        /**
         * Scatters data from the root node to all nodes in channel communicator.
         * @tparam T The type of message data to be scattered.
         * @param data The message data to be scattered.
         * @param size The number of total message elements from root node.
         * @param root The root node for scattering the messages from.
         * @param comm The communicator channel to scatter messages with.
         * @return The scattered message.
         */
        template <typename T>
        inline typename payload<T>::return_type scatter(
            T *data
          , size_t size
          , node root = museqa::node::master
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            payload<T> msg {data, size};
            return mpi::mscatter(msg, root, comm);
        }

        /**
         * Scatters a container from the root node to all nodes in channel.
         * @tparam T The type of container data to be scattered.
         * @param data The container data to be scattered.
         * @param root The root node for scattering the messages from.
         * @param comm The communicator channel to scatter messages with.
         * @return The scattered message.
         */
        template <typename T>
        inline typename payload<T>::return_type scatter(
            T& data
          , node root = museqa::node::master
          , const communicator& comm = world
        ) noexcept(!safe)
        {
            message msg = payload<T> {data};
            msg.size = mpi::broadcast(&msg.size, 1, root, comm);

            int quotient  = msg.size / comm.size;
            int remainder = msg.size % comm.size;

            if (!remainder) return mpi::mscatter(msg, root, comm);

            payload<int> size = factory::buffer<int>(comm.size);
            payload<int> disp = factory::buffer<int>(comm.size);

            for (int i = 0; i < comm.size; ++i) {
                size[i] = quotient + (remainder > i);
                disp[i] = quotient * i + utility::min(i, remainder);
            }

            return mpi::mscatterv(msg, size, disp, root, comm);
        }
    }
}

#endif
