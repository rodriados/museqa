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
    /*
     * Declaring global variables.
     */
    extern const int master = __msa_master_node_id__;

    /**
     * Static template struct responsible for informing the datatype of data
     * to be sent via MPI. It can only be used with the types defined below.
     * @since 0.1.alpha
     */
    template<typename T, typename U = void> struct type;

    #define type_entry(kvl, ...)                            \
        template<> struct type<__VA_ARGS__>                 \
        {                                                   \
            inline static const MPI_Datatype get()          \
            {                                               \
                return (kvl);                               \
            }                                               \
        };

    type_entry(MPI_CHAR, int8_t);
    type_entry(MPI_BYTE, uint8_t);
    type_entry(MPI_SHORT, int16_t);
    type_entry(MPI_UNSIGNED_SHORT, uint16_t);
    type_entry(MPI_INT, int32_t);
    type_entry(MPI_UNSIGNED, uint32_t);
    type_entry(MPI_LONG, int64_t);
    type_entry(MPI_UNSIGNED_LONG, uint64_t);

    type_entry(MPI_FLOAT, float);
    type_entry(MPI_DOUBLE, double);

    type_entry(MPI_2INT, int32_t, int32_t);
    type_entry(MPI_FLOAT_INT, float, int32_t);
    type_entry(MPI_DOUBLE_INT, double, int32_t);

    /**
     * Initializes the node's communication and identifies it in the cluster.
     * @param argc The number of arguments sent from terminal.
     * @param argv The arguments sent from terminal.
     */
    inline void init(int& argc, char **& argv)
    {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &node::rank);
        MPI_Comm_size(MPI_COMM_WORLD, &node::size);
    }
    
    /**
     * Broadcasts data to all nodes connected in the cluster.
     * @param buffer The buffer to broadcast.
     * @param count The number of buffer's elements to broadcast.
     * @param root The operation's root node.
     */
    template<typename T, typename U>
    inline int broadcast(void *buffer, int count = 1, int root = master)
    {
        return MPI_Bcast(buffer, count, type<T,U>::get(), root, MPI_COMM_WORLD);
    }

    template<typename T>
    inline int broadcast(T *buffer, int count = 1, int root = master)
    {
        return MPI_Bcast(buffer, count, type<T>::get(), root, MPI_COMM_WORLD);
    }

    /**
     * Sends data to a node connected to the cluster.
     * @param buffer The buffer to send.
     * @param count The number of buffer's elements to send.
     * @param dest The destination node.
     * @param tag The identifying tag.
     */
    template<typename T, typename U>
    inline int send(const void *buffer, int count = 1, int dest = master, int tag = MPI_TAG_UB)
    {
        return MPI_Send(buffer, count, type<T,U>::get(), dest, tag, MPI_COMM_WORLD);
    }

    template<typename T>
    inline int send(const T *buffer, int count = 1, int dest = master, int tag = MPI_TAG_UB)
    {
        return MPI_Send(buffer, count, type<T>::get(), dest, tag, MPI_COMM_WORLD);
    }

    /**
     * Receives data from a node connected to the cluster.
     * @param buffer The buffer to receive data into.
     * @param count The number of buffer's elements to receive.
     * @param source The source node.
     * @param tag The identifying tag.
     * @param status The transmission status.
     */
    template<typename T, typename U>
    inline int receive
    (   void *buffer
    ,   int count = 1
    ,   int source = master
    ,   int tag = MPI_TAG_UB
    ,   MPI_Status *status = MPI_STATUS_IGNORE
    ) {
        return MPI_Recv(buffer, count, type<T,U>::get(), source, tag, MPI_COMM_WORLD, status);
    }

    template<typename T>
    inline int receive
    (   T *buffer
    ,   int count = 1
    ,   int source = master
    ,   int tag = MPI_TAG_UB
    ,   MPI_Status *status = MPI_STATUS_IGNORE
    ) {
        return MPI_Recv(buffer, count, type<T>::get(), source, tag, MPI_COMM_WORLD, status);
    }

    /**
     * Synchronizes all of the cluster's nodes.
     */
    inline int synchronize()
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