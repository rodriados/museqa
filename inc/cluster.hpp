/** 
 * Multiple Sequence Alignment cluster header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef CLUSTER_HPP_INCLUDED
#define CLUSTER_HPP_INCLUDED

#pragma once

#include <cstdint>
#include <vector>
#include <mpi.h>

#include "node.hpp"
#include "reflection.hpp"

namespace cluster
{
    /*
     * Declaring global variable.
     */
    extern std::vector<MPI_Datatype> custom;

    /*
     * Forward declaration of Datatype, so that datatypes that already exist can
     * be used to form a new datatype.
     */
    template <typename T>
    struct Datatype;

    namespace internal
    {
        /**
         * The initial struct for generating new datatypes.
         * @tparam T The type to be represented by the datatype.
         * @since 0.1.alpha
         */
        template <typename T>
        struct GenerateDatatype
        {
            /**
             * Initializes the recursion through the reflection tuple of type so its corresponding
             * datatype is created.
             * @param blockl The list of block lengths in type T.
             * @param offset The list of field offsets in type T.
             * @param type The list of datatypes in type T.
             */
            inline static void gen(int *blockl, MPI_Aint *offset, MPI_Datatype *type)
            {
                GenerateDatatype<typename Reflection<T>::Tuple>::template gen<T>(blockl, offset, type);
            }
        };

        /**
         * The final recursion step for creating a new datatype.
         * @param T The last type in reflection tuple to become part of the datatype.
         * @since 0.1.alpha
         */
        template <typename T>
        struct GenerateDatatype<reflection::Tuple<T>>
        {
            /**
             * Terminates the recursion through the reflection tuple of the original type.
             * @tparam O The original type corresponding to the datatype.
             * @tparam N The current recurstion step.
             * @param blockl The list of block lengths in type T.
             * @param offset The list of field offsets in type T.
             * @param type The list of datatypes in type T.             
             */
            template <typename O, size_t N = 0>
            inline static void gen(int *blockl, MPI_Aint *offset, MPI_Datatype *type)
            {
                blockl[N] = 1;
                type[N] = Datatype<T>::get();
                offset[N] = Reflection<O>::template getOffset<N>();
            }
        };

        /**
         * The middle recursion steps for creating a new datatype.
         * @tparam T The current type in reflection tuple to become part of the datatype.
         * @tparam U The following types in reflection tuple.
         * @since 0.1.alpha
         */
        template <typename T, typename ...U>
        struct GenerateDatatype<reflection::Tuple<T, U...>>
        {
            /**
             * Processes a step of the recursion through the reflection tuple of the original type.
             * @tparam O The original type corresponding to the datatype.
             * @tparam N The current recurstion step.
             * @param blockl The list of block lengths in type T.
             * @param offset The list of field offsets in type T.
             * @param type The list of datatypes in type T.             
             */
            template <typename O, size_t N = 0>
            inline static void gen(int *blockl, MPI_Aint *offset, MPI_Datatype *type)
            {
                GenerateDatatype<reflection::Tuple<T>>::template gen<O, N>(blockl, offset, type);
                GenerateDatatype<reflection::Tuple<U...>>::template gen<O, N+1>(blockl, offset, type);
            }
        };
    };

    /**
     * Generates a new datatype for a user defined type.
     * @tparam T The type to which datatype must be created.
     * @since 0.1.alpha
     */
    template <typename T>
    struct Datatype
    {
        protected:
            MPI_Datatype dtypeid;       /// The datatype of type T.

        private:
            /**
             * Builds a new instance and creates the datatype for the requested type.
             * This constructor shall be called only once for each type to be created.
             */
            inline Datatype() noexcept
            {
                const int size = Reflection<T>::getSize();

                int blockl[size];
                MPI_Aint offset[size];
                MPI_Datatype type[size];

                internal::GenerateDatatype<T>::gen(blockl, offset, type);

                MPI_Type_create_struct(size, blockl, offset, type, &this->dtypeid);
                MPI_Type_commit(&this->dtypeid);

                custom.push_back(this->dtypeid);
            }

            /**
             * Recovers the instance, or creates one, that holds the created datatype of
             * the requested type.
             * @return The instance with created datatype.
             */
            inline static Datatype<T>& getInstance()
            {
                static Datatype<T> instance;
                return instance;
            }

        public:
            /**
             * Gives access to the datatype created for the requested type.
             * @return The datatype created for type T.
             */
            inline static MPI_Datatype get()
            {
                return Datatype<T>::getInstance().dtypeid;
            }
    };

    /*
     * Macro responsible for placing methods for built-in datatypes.
     */
    #define use_built_in(builtin)                               \
        inline static MPI_Datatype get()                        \
        {                                                       \
            return builtin;                                     \
        }

    /**#@+
     * Template specializations for built-in types. These types are created for built-in
     * types automatically and do not need to be created.
     * @since 0.1.alpha
     */
    template <> struct Datatype<char>     { use_built_in(MPI_CHAR); };
    template <> struct Datatype<int8_t>   { use_built_in(MPI_CHAR); };
    template <> struct Datatype<uint8_t>  { use_built_in(MPI_BYTE); };
    template <> struct Datatype<int16_t>  { use_built_in(MPI_SHORT); };
    template <> struct Datatype<uint16_t> { use_built_in(MPI_UNSIGNED_SHORT); };
    template <> struct Datatype<int32_t>  { use_built_in(MPI_INT); };
    template <> struct Datatype<uint32_t> { use_built_in(MPI_UNSIGNED); };
    template <> struct Datatype<int64_t>  { use_built_in(MPI_LONG); };
    template <> struct Datatype<uint64_t> { use_built_in(MPI_UNSIGNED_LONG); };
    template <> struct Datatype<float>    { use_built_in(MPI_FLOAT); };
    template <> struct Datatype<double>   { use_built_in(MPI_DOUBLE); };
    /**#@-*/
    #undef use_built_in

    /**
     * Initializes the node's communication and identifies it in the cluster.
     * @param argc The number of arguments sent from terminal.
     * @param argv The arguments sent from terminal.
     */
    inline void init(int& argc, char **& argv)
    {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &cluster::rank);
        MPI_Comm_size(MPI_COMM_WORLD, &cluster::size);
    }

    /**
     * Broadcasts data to all nodes connected in the cluster.
     * @tparam T Type of buffer data to broadcast.
     * @param buffer The buffer to broadcast.
     * @param count The number of buffer's elements to broadcast.
     * @param root The operation's root node.
     * @return MPI error code if not successful.
     */
    template <typename T>
    inline int broadcast
        (   T *buffer
        ,   int count = 1
        ,   int root = master   )
    {
        return MPI_Bcast(buffer, count, Datatype<T>::get(), root, MPI_COMM_WORLD);
    }

    /**
     * Sends data to a node connected to the cluster.
     * @tparam T Type of buffer data to send.
     * @param buffer The buffer to send.
     * @param count The number of buffer's elements to send.
     * @param dest The destination node.
     * @param tag The identifying tag.
     * @return MPI error code if not successful.
     */
    template <typename T>
    inline int send
        (   const T *buffer
        ,   int count = 1
        ,   int dest = master
        ,   int tag = MPI_TAG_UB    )
    {
        return MPI_Send(buffer, count, Datatype<T>::get(), dest, tag, MPI_COMM_WORLD);
    }

    /**
     * Receives data from a node connected to the cluster.
     * @tparam T Type of buffer data to receive.
     * @param buffer The buffer to receive data into.
     * @param count The number of buffer's elements to receive.
     * @param source The source node.
     * @param tag The identifying tag.
     * @param status The transmission status.
     * @return MPI error code if not successful.
     */
    template <typename T>
    inline int receive
        (   T *buffer
        ,   int count = 1
        ,   int source = master
        ,   int tag = MPI_TAG_UB
        ,   MPI_Status *status = MPI_STATUS_IGNORE  )
    {
        return MPI_Recv(buffer, count, Datatype<T>::get(), source, tag, MPI_COMM_WORLD, status);
    }

    /**
     * Synchronizes all of the cluster's nodes.
     * @return MPI error code if not successful.
     */
    inline int sync()
    {
        return MPI_Barrier(MPI_COMM_WORLD);
    }

    /**
     * Finalizes the node's communication to the cluster.
     * @return MPI error code if not successful.
     */
    inline int finalize()
    {
        for(MPI_Datatype& dtype : custom)
            MPI_Type_free(&dtype);

        return MPI_Finalize();
    }
};

#endif