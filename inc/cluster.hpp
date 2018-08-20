/** 
 * Multiple Sequence Alignment cluster header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _CLUSTER_HPP_
#define _CLUSTER_HPP_

#include <mpi.h>

#include "node.hpp"

namespace cluster
{
    /**
     * Static template struct responsible for informing the datatype of data
     * to be sent via MPI. It can only be used with the types defined below.
     * @since 0.1.alpha
     */
    template <typename T, typename U = void> struct Datatype;
    #  define __typeid(dtype) static constexpr MPI_Datatype id = dtype

    template <> struct Datatype<char>     { __typeid(MPI_CHAR); };
    template <> struct Datatype<int8_t>   { __typeid(MPI_CHAR); };
    template <> struct Datatype<uint8_t>  { __typeid(MPI_BYTE); };
    template <> struct Datatype<int16_t>  { __typeid(MPI_SHORT); };
    template <> struct Datatype<uint16_t> { __typeid(MPI_UNSIGNED_SHORT); };
    template <> struct Datatype<int32_t>  { __typeid(MPI_INT); };
    template <> struct Datatype<uint32_t> { __typeid(MPI_UNSIGNED); };
    template <> struct Datatype<int64_t>  { __typeid(MPI_LONG); };
    template <> struct Datatype<uint64_t> { __typeid(MPI_UNSIGNED_LONG); };
    template <> struct Datatype<float>    { __typeid(MPI_FLOAT); };
    template <> struct Datatype<double>   { __typeid(MPI_DOUBLE); };

    template <> struct Datatype<int32_t, int32_t> { __typeid(MPI_2INT); };
    template <> struct Datatype<float, int32_t>   { __typeid(MPI_FLOAT_INT); };
    template <> struct Datatype<double, int32_t>  { __typeid(MPI_DOUBLE_INT); };

    /**
     * Initializes the node's communication and identifies it in the cluster.
     * @param argc The number of arguments sent from terminal.
     * @param argv The arguments sent from terminal.
     */
    inline void init(int& argc, char **& argv)
    {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &node::rank);
        MPI_Comm_size(MPI_COMM_WORLD, &cluster::size);
    }
    
    /**
     * Broadcasts data to all nodes connected in the cluster.
     * @param buffer The buffer to broadcast.
     * @param count The number of buffer's elements to broadcast.
     * @param root The operation's root node.
     */
    template <typename T, typename U>
    inline int broadcast
        (   void *buffer
        ,   int count = 1
        ,   int root = master
        )
    {
        return MPI_Bcast(buffer, count, Datatype<T,U>::id, root, MPI_COMM_WORLD);
    }

    /**
     * Broadcasts data to all nodes connected in the cluster.
     * @param buffer The buffer to broadcast.
     * @param count The number of buffer's elements to broadcast.
     * @param root The operation's root node.
     */
    template <typename T>
    inline int broadcast
        (   T *buffer
        ,   int count = 1
        ,   int root = master
        )
    {
        return MPI_Bcast(buffer, count, Datatype<T>::id, root, MPI_COMM_WORLD);
    }

    /**
     * Sends data to a node connected to the cluster.
     * @param buffer The buffer to send.
     * @param count The number of buffer's elements to send.
     * @param dest The destination node.
     * @param tag The identifying tag.
     */
    template <typename T, typename U>
    inline int send
        (   const void *buffer
        ,   int count = 1
        ,   int dest = master
        ,   int tag = MPI_TAG_UB
        )
    {
        return MPI_Send(buffer, count, Datatype<T,U>::id, dest, tag, MPI_COMM_WORLD);
    }

    /**
     * Sends data to a node connected to the cluster.
     * @param buffer The buffer to send.
     * @param count The number of buffer's elements to send.
     * @param dest The destination node.
     * @param tag The identifying tag.
     */
    template <typename T>
    inline int send
        (   const T *buffer
        ,   int count = 1
        ,   int dest = master
        ,   int tag = MPI_TAG_UB
        )
    {
        return MPI_Send(buffer, count, Datatype<T>::id, dest, tag, MPI_COMM_WORLD);
    }

    /**
     * Receives data from a node connected to the cluster.
     * @param buffer The buffer to receive data into.
     * @param count The number of buffer's elements to receive.
     * @param source The source node.
     * @param tag The identifying tag.
     * @param status The transmission status.
     */
    template <typename T, typename U>
    inline int receive
        (   void *buffer
        ,   int count = 1
        ,   int source = master
        ,   int tag = MPI_TAG_UB
        ,   MPI_Status *status = MPI_STATUS_IGNORE
        )
    {
        return MPI_Recv(buffer, count, Datatype<T,U>::id, source, tag, MPI_COMM_WORLD, status);
    }

    /**
     * Receives data from a node connected to the cluster.
     * @param buffer The buffer to receive data into.
     * @param count The number of buffer's elements to receive.
     * @param source The source node.
     * @param tag The identifying tag.
     * @param status The transmission status.
     */
    template <typename T>
    inline int receive
        (   T *buffer
        ,   int count = 1
        ,   int source = master
        ,   int tag = MPI_TAG_UB
        ,   MPI_Status *status = MPI_STATUS_IGNORE
        )
    {
        return MPI_Recv(buffer, count, Datatype<T>::id, source, tag, MPI_COMM_WORLD, status);
    }

    /**
     * Synchronizes all of the cluster's nodes.
     */
    inline int sync()
    {
        return MPI_Barrier(MPI_COMM_WORLD);
    }

    /**
     * Finalizes the node's communication to the cluster.
     */
    inline int finalize()
    {
        return MPI_Finalize();
    }
};

#endif