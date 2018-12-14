/** 
 * Multiple Sequence Alignment cluster header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef CLUSTER_HPP_INCLUDED
#define CLUSTER_HPP_INCLUDED

#pragma once

#ifndef msa_compile_cython

#include <cstdint>
#include <vector>
#include <mpi.h>

#include "node.hpp"
#include "buffer.hpp"
#include "reflection.hpp"

namespace cluster
{
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
         * @since 0.1.1
         */
        template <typename T>
        struct DatatypeBuilder
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
                DatatypeBuilder<typename Reflection<T>::Tuple>::template gen<T>(blockl, offset, type);
            }
        };

        /**
         * The final recursion step for creating a new datatype.
         * @param T The last type in reflection tuple to become part of the datatype.
         * @since 0.1.1
         */
        template <typename T>
        struct DatatypeBuilder<reflection::Tuple<T>>
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
         * @since 0.1.1
         */
        template <typename T, typename ...U>
        struct DatatypeBuilder<reflection::Tuple<T, U...>>
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
                DatatypeBuilder<reflection::Tuple<T>>::template gen<O, N>(blockl, offset, type);
                DatatypeBuilder<reflection::Tuple<U...>>::template gen<O, N+1>(blockl, offset, type);
            }
        };
    };

    extern std::vector<MPI_Datatype> dtypes;
    static const MPI_Comm world = MPI_COMM_WORLD;

    /**
     * Generates a new datatype for a user defined type.
     * @tparam T The type to which datatype must be created.
     * @since 0.1.1
     */
    template <typename T>
    struct Datatype
    {
        protected:
            MPI_Datatype dtype;       /// The datatype of type T.

        private:
            /**
             * Builds a new instance and creates the datatype for the requested type.
             * This constructor shall be called only once for each type to be created.
             */
            inline Datatype() noexcept
            {
                constexpr int size = Reflection<T>::getSize();

                int blockl[size];
                MPI_Aint offset[size];
                MPI_Datatype type[size];

                internal::DatatypeBuilder<T>::gen(blockl, offset, type);

                MPI_Type_create_struct(size, blockl, offset, type, &dtype);
                MPI_Type_commit(&dtype);

                dtypes.push_back(dtype);
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
                return Datatype<T>::getInstance().dtype;
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
     * @since 0.1.1
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
     * Represents the base of a communication payload.
     * @tparam T The payload's data type.
     * @since 0.1.1
     */
    template <typename T>
    class BasePayload
    {
        protected:
            T *buffer = nullptr;    /// The payload's buffer.
            bool dynamic = false;   /// Is the payload dynamic?
            size_t size = 0;        /// The payload's buffer's size.

        public:
            BasePayload() = delete;
            BasePayload(const BasePayload<T>&) = delete;
            BasePayload(BasePayload<T>&&) = delete;

            /**
             * Creates a new payload from buffer.
             * @param buffer The payload's buffer.
             */
            BasePayload(T& buffer)
            :   buffer(&buffer)
            ,   size(1) {}

            /**
             * Creates a new payload from buffer.
             * @param buffer The payload's buffer.
             * @param size The buffer's size.
             */
            BasePayload(T *buffer, size_t size = 1)
            :   buffer(buffer)
            ,   size(size) {}

            BasePayload<T>& operator=(const BasePayload<T>&) = delete;
            BasePayload<T>& operator=(BasePayload<T>&&) = delete;

            /**
             * Broadcasts data to all nodes connected in the cluster.
             * @param root The operation's root node.
             * @return MPI error code if not successful.
             */
            inline int broadcast(int root = master)
            {
                if(dynamic) {
                    MPI_Bcast(&size, 1, Datatype<size_t>::get(), root, world);
                    if(cluster::rank != root) resize(size);
                }

                return MPI_Bcast(buffer, size, Datatype<T>::get(), root, world);
            }

            /**
             * Receives data from a node connected to the cluster.
             * @param source The source node.
             * @param tag The identifying tag.
             * @param status The transmission status.
             * @return MPI error code if not successful.
             */
            inline int receive(int source = master, int tag = MPI_TAG_UB, MPI_Status *status = MPI_STATUS_IGNORE)
            {
                if(dynamic) {
                    int size;
                    MPI_Status probed;
                    MPI_Probe(source, tag, world, &probed);
                    MPI_Get_count(&probed, Datatype<int>::get(), &size);
                    resize(size);
                }

                return MPI_Recv(buffer, size, Datatype<T>::get(), source, tag, world, status);
            }

            /**
             * Sends data to a node connected to the cluster.
             * @param dest The destination node.
             * @param tag The identifying tag.
             * @return MPI error code if not successful.
             */
            inline int send(int dest = master, int tag = MPI_TAG_UB)
            {
                return MPI_Send(buffer, size, Datatype<T>::get(), dest, tag, world);
            }

            /**
             * Scatters data to nodes according to given distribution.
             * @param buffer The buffer to be scattered.
             * @param count The count of elements to send to each node.
             * @param displ The scatter displacement of each node.
             * @param root The operation's root node.
             * @return MPI error code if not successful.
             */
            inline int scatter(BasePayload<T>& buffer, int *count, int *displ, int root = master)
            {
                if(dynamic) {
                    resize(count[rank]);
                }

                return MPI_Scatterv(
                    buffer.buffer, count, displ, Datatype<T>::get(),
                    this->buffer, count[rank], Datatype<T>::get(), root, world
                );
            }

        protected:
            /**
             * Creates payload from buffer. Internal constructor.
             * @param buffer The payload's buffer.
             * @param size The buffer's size.
             * @param dynamic Is the payload dynamic?
             */
            BasePayload(T *buffer, size_t size, bool dynamic = false)
            :   buffer(buffer)
            ,   dynamic(dynamic)
            ,   size(size) {}

            /**
             * Resizes the dynamic payload so it can store all received data.
             * @param size The new payload size.
             */
            virtual void resize(size_t size)
            {
                this->size = size;
            }
    };

    /**
     * Represents a communication payload.
     * @tparam T The payload's data type.
     * @since 0.1.1
     * @see P0136R1
     */
    template <typename T>
    class Payload : public BasePayload<T>
    {
        // Due to compilter defect report P0136R1, the base class constructor cannot be
        // inherited with "using", as it may inject additional constructors in the derived
        // class. For this reason, the base constructors are explicitly overriden here.

        public:
            /**
             * Creates a new payload from buffer.
             * @param buffer The payload's buffer.
             */
            Payload(T& buffer)
            :   BasePayload<T>(buffer) {}

            /**
             * Creates a new payload from buffer.
             * @param buffer The payload's buffer.
             * @param size The buffer's size.
             */
            Payload(T *buffer, size_t size = 1)
            :   BasePayload<T>(buffer, size) {}
    };

    /**
     * Represents a communication payload.
     * @tparam T The payload's data type.
     * @since 0.1.1
     */
    template <typename T>
    class Payload<std::vector<T>> : public BasePayload<T>
    {
        protected:
            std::vector<T>& target; /// The original target vector.

        public:
            /**
             * Creates a new payload from vector.
             * @param vector The payload's vector.
             */
            Payload(std::vector<T>& vector)
            :   BasePayload<T>(vector.data(), vector.size(), true)
            ,   target(vector) {}

        protected:
            /**
             * Resizes the dynamic payload so it can store all received data.
             * @param size The new payload size.
             */
            void resize(size_t size) override
            {
                target.resize(size);
                this->buffer = target.data();
                this->size = size;
            }
    };

    /**
     * Represents a communication payload.
     * @tparam T The payload's data type.
     * @since 0.1.1
     */
    template <typename T>
    class Payload<BaseBuffer<T>> : public BasePayload<T>
    {
        protected:
            BaseBuffer<T>& target;  /// The original target buffer.

        public:
            /**
             * Creates a new payload from buffer.
             * @param buffer The payload's buffer.
             */
            Payload(BaseBuffer<T>& buffer)
            :   BasePayload<T>(buffer.getBuffer(), buffer.getSize(), true)
            ,   target(buffer) {}

        protected:
            /**
             * Resizes the dynamic payload so it can store all received data.
             * @param size The new payload size.
             */
            void resize(size_t size) override
            {
                target = {new T[size], size};
                this->buffer = target.getBuffer();
                this->size = size;
            }
    };

    /**
     * Initializes the node's communication and identifies it in the cluster.
     * @param argc The number of arguments sent from terminal.
     * @param argv The arguments sent from terminal.
     */
    inline void init(int& argc, char **& argv)
    {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(world, &cluster::rank);
        MPI_Comm_size(world, &cluster::size);
    }

    /**#@+
     * Broadcasts data to all nodes connected in the cluster.
     * @tparam T Type of buffer data to broadcast.
     * @param buffer The buffer to broadcast.
     * @param count The number of buffer's elements to broadcast.
     * @param root The operation's root node.
     * @return MPI error code if not successful.
     */
    template <typename T>
    inline int broadcast
        (   T& buffer
        ,   int root = master               )
    {
        Payload<T> payload(buffer);
        return payload.broadcast(root);
    }

    template <typename T>
    inline int broadcast
        (   T *buffer
        ,   int count = 1
        ,   int root = master               )
    {
        Payload<T> payload(buffer, count);
        return payload.broadcast(root);
    }
    /**#@-*/

    /**#@+
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
        (   T& buffer
        ,   int source = master
        ,   int tag = MPI_TAG_UB
        ,   MPI_Status *status = MPI_STATUS_IGNORE  )
    {
        Payload<T> payload(buffer);
        return payload.receive(source, tag, status);
    }

    template <typename T>
    inline int receive
        (   T *buffer
        ,   int count = 1
        ,   int source = master
        ,   int tag = MPI_TAG_UB
        ,   MPI_Status *status = MPI_STATUS_IGNORE  )
    {
        Payload<T> payload(buffer, count);
        return payload.receive(source, tag, status);
    }
    /**#@-*/

    /**#@+
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
        (   T& buffer
        ,   int dest = master
        ,   int tag = MPI_TAG_UB            )
    {
        Payload<T> payload(buffer);
        return payload.send(dest, tag);
    }

    template <typename T>
    inline int send
        (   T *buffer
        ,   int count = 1
        ,   int dest = master
        ,   int tag = MPI_TAG_UB            )
    {
        Payload<T> payload(buffer, count);
        return payload.send(dest, tag);
    }
    /**#@-*/

    /**#@+
     * Scatters data to nodes according to given distribution.
     * @tparam T The type of buffer data to scatter.
     * @param sendbuf The buffer to be scattered.
     * @param recvbuf The receiving buffer.
     * @param sendcount The count of elements to send to each node.
     * @param senddispl The scatter displacement of each node.
     * @param root The operation's root node.
     * @return MPI error code if not successful.
     */
    template <typename T>
    inline int scatter
        (   T& sendbuf
        ,   T& recvbuf
        ,   std::vector<int>& sendcount
        ,   std::vector<int>& senddispl
        ,   int root = master               )
    {
        Payload<T> payload(recvbuf), buffer(sendbuf);
        return payload.scatter(buffer, sendcount.data(), senddispl.data(), root);
    }

    template <typename T>
    inline int scatter
        (   T *sendbuf
        ,   T *recvbuf
        ,   int *sendcount
        ,   int *senddispl
        ,   int root = master               )
    {
        Payload<T> payload(recvbuf, size), buffer(sendbuf);
        return payload.scatter(buffer, sendcount, senddispl, root);
    }
    /**#@-*/

    /**
     * Synchronizes all of the cluster's nodes.
     * @return MPI error code if not successful.
     * @see device::sync
     */
    inline int sync()
    {
        return MPI_Barrier(world);
    }

    /**
     * Finalizes the node's communication to the cluster.
     * @return MPI error code if not successful.
     */
    inline int finalize()
    {
        for(MPI_Datatype& dtype : dtypes)
            MPI_Type_free(&dtype);

        return MPI_Finalize();
    }
};

#endif
#endif