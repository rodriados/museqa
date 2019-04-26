/** 
 * Multiple Sequence Alignment MPI wrapper header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef MPI_WRAPPER_INCLUDED
#define MPI_WRAPPER_INCLUDED

#ifndef msa_compile_cython

#include <mpi.h>
#include <vector>
#include <string>
#include <utility>
#include <cstdint>
#include <numeric>
#include <cstdlib>
#include <algorithm>

#include "node.hpp"
#include "tuple.hpp"
#include "utils.hpp"
#include "buffer.hpp"
#include "exception.hpp"
#include "reflection.hpp"

namespace mpi
{
    /**
     * Represents the identifier of a node connected to the cluster.
     * @since 0.1.1
     */
    using Node = int32_t;

    /**
     * Represents a message tag, used for message identification and encapsulation.
     * @since 0.1.1
     */
    using Tag = int32_t;

    /**
     * Permits communication and synchronization among a set of nodes and processes.
     * @since 0.1.1
     */
    struct Communicator
    {
        Node rank = 0;                  /// The rank of current node according in the communicator.
        int size = 0;                   /// The number of nodes in the communicator.
        MPI_Comm id = MPI_COMM_NULL;    /// The communicator identifier.
    };

    /**
     * The default communicator instance.
     * @see mpi::Communicator
     */
    extern Communicator world;

    /**
     * The value acceptable as a node source and a tag for a received message.
     * @since 0.1.1
     */
    enum : int16_t { any = -1 };

    namespace error
    {
        /**
         * Produces an error message explaining error obtained by code.
         * @param code The error code to be explained.
         * @return The error description.
         */
        inline std::string describe(int code)
        {
            int length;
            char buffer[2048];

            return MPI_Error_string(code, buffer, &length) != MPI_SUCCESS
                ? "error while probing MPI error"
                : buffer;
        }
    }

    /**
     * Holds a MPI error message so it can be propagated through the code.
     * @since 0.1.1
     */
    struct Exception : public ::Exception
    {
        int code;               /// The failed result code reported by MPI implementation.

        /**
         * Builds a new exception instance.
         * @param code The result code reported by MPI.
         */
        inline Exception(int code)
        :   ::Exception {"MPI Exception:", error::describe(code)}
        ,   code {code}
        {}

        /**
         * Builds a new exception instance from error code.
         * @tparam P The format parameters' types.
         * @param code The error code.
         * @param fmt The additional message's format.
         * @param args The format's parameters.
         */
        template <typename ...P>
        inline Exception(int code, const std::string& fmt, P... args)
        :   ::Exception {"MPI Exception:", error::describe(code), args...}
        ,   code {code}
        {}

        /**
         * Retrieves the MPI error code thrown.
         * @param The error code.
         */
        inline int getCode() const
        {
            return code;
        }
    };

    /**
     * Checks whether a MPI operation has been successful and throws error if not.
     * @tparam P The format string parameter types.
     * @param code The error code obtained from the operation.
     * @param fmt The format string to use as error message.
     * @param args The format string values.
     * @throw The error code obtained raised to exception.
     */
    template <typename ...P>
    inline void call(int code, const std::string& fmt = {}, P... args)
    {
        if(code != MPI_SUCCESS)
            throw Exception {code, fmt, args...};
    }

    namespace datatype
    {
        template <typename T>
        class Generator;

        /**
         * Gives access to the datatype created for the requested type.
         * @tparam T The requested type.
         * @return The datatype created for type T.
         */
        template <typename T>
        inline MPI_Datatype get() { return Generator<T>::get(); }

        /**#@+
         * Template specializations for built-in types. These types are created for built-in
         * types automatically and, thus, can be used directly.
         * @since 0.1.1
         */
        template <> inline MPI_Datatype get<bool>()     { return MPI_C_BOOL; };
        template <> inline MPI_Datatype get<float>()    { return MPI_FLOAT; };
        template <> inline MPI_Datatype get<double>()   { return MPI_DOUBLE; };
        template <> inline MPI_Datatype get<int8_t>()   { return MPI_INT8_T; };
        template <> inline MPI_Datatype get<uint8_t>()  { return MPI_UINT8_T; };
        template <> inline MPI_Datatype get<int16_t>()  { return MPI_INT16_T; };
        template <> inline MPI_Datatype get<uint16_t>() { return MPI_UINT16_T; };
        template <> inline MPI_Datatype get<int32_t>()  { return MPI_INT32_T; };
        template <> inline MPI_Datatype get<uint32_t>() { return MPI_UINT32_T; };
        template <> inline MPI_Datatype get<int64_t>()  { return MPI_INT64_T; };
        template <> inline MPI_Datatype get<uint64_t>() { return MPI_UINT64_T; };
        /**#@-*/

        namespace detail
        {
            /**
             * The initial step for generating new datatypes.
             * @tparam T The type to be represented by the datatype.
             * @since 0.1.1
             */
            template <typename T>
            struct Builder
            {
                /**
                 * Initializes the recursion through the type's reflection tuple.
                 * @param blockList The list of block lengths in type T.
                 * @param offsetList The list of field offsets in type T.
                 * @param typeList The list of datatypes in type T.
                 */
                inline static void generate(int *blockList, MPI_Aint *offsetList, MPI_Datatype *typeList)
                {
                    Builder<typename Reflection<T>::Tuple>::template generate<T>(blockList, offsetList, typeList);
                }
            };

            /**
             * The final recursion step for creating a new datatype.
             * @param T The last type in reflection tuple to become part of the datatype.
             * @since 0.1.1
             */
            template <typename T>
            struct Builder<Tuple<T>>
            {
                /**
                 * Terminates the recursion through the original type's reflection tuple.
                 * @tparam O The original type corresponding to the datatype.
                 * @tparam N The current recurstion step.
                 * @param blockList The list of block lengths in type T.
                 * @param offsetList The list of field offsets in type T.
                 * @param typeList The list of datatypes in type T.
                 */
                template <typename O, size_t N = 0>
                inline static void generate(int *blockList, MPI_Aint *offsetList, MPI_Datatype *typeList)
                {
                    blockList[N] = 1;
                    typeList[N] = get<T>();
                    offsetList[N] = Reflection<O>::template getOffset<N>();
                }
            };

            /**
             * The middle recursion steps for creating a new datatype.
             * @tparam T The current type in reflection tuple to become part of the datatype.
             * @tparam U The following types in reflection tuple.
             * @since 0.1.1
             */
            template <typename T, typename ...U>
            struct Builder<Tuple<T, U...>>
            {
                /**
                 * Processes a step of the recursion through the original type's reflection tuple.
                 * @tparam O The original type corresponding to the datatype.
                 * @tparam N The current recurstion step.
                 * @param blockList The list of block lengths in type T.
                 * @param offsetList The list of field offsets in type T.
                 * @param typeList The list of datatypes in type T.             
                 */
                template <typename O, size_t N = 0>
                inline static void generate(int *blockList, MPI_Aint *offsetList, MPI_Datatype *typeList)
                {
                    Builder<Tuple<T>>::template generate<O, N>(blockList, offsetList, typeList);
                    Builder<Tuple<U...>>::template generate<O, N+1>(blockList, offsetList, typeList);
                }
            };
        };

        /**
         * Keeps track of all generated datatypes throughout execution.
         * @since 0.1.1
         */
        extern std::vector<MPI_Datatype> dtypes;

        /**
         * Generates a new datatype for a user defined type.
         * @tparam T The type to which datatype must be created.
         * @since 0.1.1
         */
        template <typename T>
        struct Generator
        {
            MPI_Datatype typeId;

            static_assert(!std::is_union<T>::value, "Unions should not be messaged via MPI.");

            /**
             * Builds a new instance and creates the datatype for the requested type.
             * This constructor shall be called only once for each type to be created.
             * @see Generator::get
             */
            inline Generator() noexcept
            {
                constexpr const int size = Reflection<T>::getSize();

                int blockList[size];
                MPI_Aint offsetList[size];
                MPI_Datatype typeList[size];

                detail::Builder<T>::generate(blockList, offsetList, typeList);

                MPI_Type_create_struct(size, blockList, offsetList, typeList, &typeId);
                MPI_Type_commit(&typeId);

                dtypes.push_back(typeId);
            }

            /**
             * Gives access to the datatype created for the requested type.
             * @return The datatype created for type T.
             */
            inline static MPI_Datatype get()
            {
                static Generator<T> instance;
                return instance.typeId;
            }
        };
    };

    /**
     * Contains information about a message that has been or can be received.
     * @since 0.1.1
     */
    struct Status
    {
        mutable MPI_Status status;

        Status() = default;
        Status(const Status&) = default;

        /**
         * Instatiates a new status object.
         * @param status The MPI status built-in object.
         */
        inline Status(const MPI_Status& status)
        :   status {status}
        {}

        /**
         * Converts to the built-in status object.
         * @return The built-in status object.
         */
        inline operator MPI_Status&()
        {
            return status;
        }

        /**
         * Retrieves the message error code.
         * @return The error code.
         */
        inline int getError() const
        {
            return status.MPI_ERROR;
        }

        /**
         * Retrieves the source of the message.
         * @return The message source node.
         */
        inline Node getSource() const
        {
            return status.MPI_SOURCE;
        }

        /**
         * Retrieves the message tag.
         * @return The retrieved message tag.
         */
        inline Tag getTag() const
        {
            return status.MPI_TAG;
        }

        /**
         * Determines the number of elements contained in the message.
         * @tparam T The message content type.
         * @return The number of elements contained in the message.
         */
        template <typename T>
        inline size_t getCount() const
        {
            int value;
            MPI_Get_count(&status, datatype::get<T>(), &value);
            return value != MPI_UNDEFINED ? value : -1;
        }

        /**
         * Determines whether the communication associated with this object
         * has been successfully cancelled.
         * @return Has the message been cancelled?
         */
        inline bool isCancelled() const
        {
            int flag = 0;
            MPI_Test_cancelled(&status, &flag);
            return flag != 0;
        }
    };

    namespace detail
    {
        /**
         * Represents incoming and outcoming message payload of communication operations.
         * In practice, this object serves as a neutral context state for messages.
         * @tparam T The message payload types.
         * @since 0.1.1
         */
        template <typename T, typename = void>
        struct Payload
        {
            using type = Pure<T>;           /// Exposes the payload's message type.

            Pure<T> *buffer = nullptr;      /// The payload's source or destiny pointer.
            size_t size = 0;                /// The payload's size.

            Payload() = default;
            Payload(const Payload<T>&) = delete;
            Payload(Payload<T>&&) = delete;

            /**
             * Creates a new payload from simple object value.
             * @param value The payload's value.
             */
            inline Payload(T& value)
            :   buffer {&value}
            ,   size {1}
            {}

            /**
             * Creates a new payload from already existing buffer.
             * @param ptr The payload's buffer pointer.
             * @param size The payload's buffer size.
             */
            inline Payload(Pure<T> *ptr, size_t size = 1)
            :   buffer {ptr}
            ,   size {size}
            {}

            Payload<T>& operator=(const Payload<T>&) = delete;
            Payload<T>& operator=(Payload<T>&&) = delete;

            /**
             * Retrieves the pointer to payload's buffer.
             * @return The payload's buffer pointer.
             */
            inline Pure<T> *getBuffer() const
            {
                return buffer;
            }

             /**
             * Retrieves the payload's buffer capacity.
             * @return The payload's size or capacity.
             */
            inline size_t getSize() const
            {
                return size;
            }

            /**
             * Allows the payload buffer to be resized, so a message of given size can be
             * successfully received.
             * @param (ignored) The new payload capacity.
             * @return The resized buffer.
             */
            inline Pure<T> *resize(size_t)
            {
                return buffer;
            }
        };

        /**
         * Message payload context for STL vectors.
         * @tparam T The message payload type.
         * @since 0.1.1
         */
        template <typename T>
        struct Payload<std::vector<T>> : public Payload<T>
        {
            std::vector<T>& object;         /// The original vector.

            /**
             * Creates a new payload from STL vector.
             * @param vector The payload as a STL vector.
             */
            inline Payload(std::vector<T>& vector)
            :   Payload<T> {vector.data(), vector.size()}
            ,   object {vector}
            {}

            /**
             * Allows the payload buffer to be resized, so a message of given size can be
             * successfully received.
             * @param size The new payload capacity.
             * @return The resized buffer.
             */
            inline Pure<T> *resize(size_t size)
            {
                if(this->size < size) object.resize(size);

                this->size = size;
                return this->buffer = object.data();
            }
        };

        /**
         * Message payload context for buffers.
         * @tparam T The message payload type.
         * @since 0.1.1
         */
        template <template <typename> class B, typename T>
        struct Payload<B<T>, typename std::enable_if<std::is_base_of<BaseBuffer<T>, B<T>>::value>::type>
            : public Payload<T>
        {
            BaseBuffer<T>& object;          /// The original buffer.

            /**
             * Creates a new payload from buffer.
             * @param buffer The buffer to use as payload.
             */
            inline Payload(BaseBuffer<T>& buffer)
            :   Payload<T> {buffer.getBuffer(), buffer.getSize()}
            ,   object {buffer}
            {}

            /**
             * Allows the payload buffer to be resized, so a message of given size can be
             * successfully received.
             * @param size The new payload capacity.
             * @return The resized buffer.
             */
            inline Pure<T> *resize(size_t size)
            {
                if(this->size < size) object = BaseBuffer<T> {size};

                this->size = size;
                return this->buffer = object.getBuffer();
            }
        };
    };

    namespace communicator
    {
        /**
         * Represents a false, non-existent or invalid communicator.
         * @since 0.1.1
         */
        static constexpr Communicator null;

        namespace detail
        {
            /**
             * Builds up a new communicator instance from built-in type.
             * @param comm Built-in communicator instance.
             * @return The new communicator instance.
             */
            inline Communicator build(MPI_Comm comm)
            {
                int rank, size;
                call(MPI_Comm_rank(comm, &rank));
                call(MPI_Comm_size(comm, &size));
                return {rank, size, comm};
            }
        };

        /**
         * Clones the communicator, creating a new communicator as a copy.
         * @param comm The communicator to clone.
         * @return The clone created from original communicator.
         */
        inline Communicator clone(const Communicator& comm)
        {
            MPI_Comm newcomm;
            call(MPI_Comm_dup(comm.id, &newcomm));
            call(MPI_Comm_set_errhandler(newcomm, MPI_ERRORS_RETURN));
            return detail::build(newcomm);
        }

        /**#@+
         * Splits nodes into different communicators according to selected color.
         * @param comm The original communicator to be split.
         * @param color The color selected by current node.
         * @param key The key used to assigned a node id in new communicator.
         * @return The obtained communicator from split operation.
         */
        inline Communicator split(const Communicator& comm, int color, int key)
        {
            MPI_Comm newcomm;
            call(MPI_Comm_split(comm.id, color, key, &newcomm));
            return detail::build(newcomm);
        }

        inline Communicator split(const Communicator& comm, int color)
        {
            return split(comm, color, comm.rank);
        }
        /**#@-*/

        /**
         * Cleans up resources used by communicator.
         * @param comm The communicator to be destroyed.
         */
        inline void free(Communicator& comm)
        {
            call(MPI_Comm_free(&comm.id));
            comm = null;
        }
    };

    /**
     * Initializes the cluster's communication and identifies the node in the cluster.
     * @param argc The number of arguments sent from terminal.
     * @param argv The arguments sent from terminal.
     */
    inline void init(int& argc, char **& argv)
    {
        call(MPI_Init(&argc, &argv));
        world = communicator::detail::build(MPI_COMM_WORLD);
        
        node::rank = world.rank;
        node::size = world.size;
    }

    /**#@+
     * Broadcasts data to all nodes in given communicator.
     * @tparam T Type of buffer data to broadcast.
     * @param buffer The buffer data to broadcast.
     * @param count The number of buffer's elements to broadcast.
     * @param comm The communicator this operation applies to.
     * @param root The operation's root node.
     */
    template <typename T>
    inline void broadcast
        (   T *buffer
        ,   int count = 1
        ,   const Node& root = node::master
        ,   const Communicator& comm = world    )
    {
        call(MPI_Bcast(buffer, count, datatype::get<T>(), root, comm.id));
    }

    template <typename T>
    inline void broadcast
        (   T& buffer
        ,   const Node& root = node::master
        ,   const Communicator& comm = world    )
    {
        detail::Payload<T> payload {buffer};
        int size = payload.getSize();

        broadcast(&size, 1, root, comm);        
        broadcast(payload.resize(size), size, root, comm);
    }
    /**#@-*/

    /**
     * Inspects incoming message and retrieves its status.
     * @param source The source node.
     * @param tag The identifying message tag.
     * @param comm The communicator this operation applies to.
     * @return The inspected message status.
     */
    inline Status probe(const Node& source = node::master, const Tag& tag = any, const Communicator& comm = world)
    {
        MPI_Status status;
        call(MPI_Probe(source, tag < 0 ? MPI_TAG_UB : tag, comm.id, &status));

        return {status};
    }

    /**#@+
     * Sends data to a node connected to the cluster.
     * @tparam T Type of buffer data to send.
     * @param buffer The buffer to send.
     * @param count The number of buffer's elements to send.
     * @param dest The destination node.
     * @param tag The identifying message tag.
     * @param comm The communicator this operation applies to.
     * @return MPI error code if not successful.
     */
    template <typename T>
    inline void send
        (   T *buffer
        ,   int count = 1
        ,   const Node& dest = node::master
        ,   const Tag& tag = MPI_TAG_UB
        ,   const Communicator& comm = world    )
    {
        call(MPI_Send(buffer, count, datatype::get<T>(), dest, tag < 0 ? MPI_TAG_UB : tag, comm.id));
    }

    template <typename T>
    inline void send
        (   T& buffer
        ,   const Node& dest = node::master
        ,   const Tag& tag = MPI_TAG_UB
        ,   const Communicator& comm = world )
    {
        detail::Payload<T> payload {buffer};
        send(payload.getBuffer(), payload.getSize(), dest, tag, comm);
    }
    /**#@-*/

    /**#@+
     * Receives data from a node connected to the cluster.
     * @tparam T Type of buffer data to receive.
     * @param buffer The buffer to receive data into.
     * @param count The number of buffer's elements to receive.
     * @param source The source node.
     * @param tag The identifying tag.
     * @param comm The communicator this operation applies to.
     * @return The message status
     */
    template <typename T>
    inline Status receive
        (   T *buffer
        ,   int count = 1
        ,   const Node& source = any
        ,   const Tag& tag = MPI_TAG_UB
        ,   const Communicator& comm = world    )
    {
        MPI_Status status;
        call(MPI_Recv(buffer, count, datatype::get<T>(), source, tag, comm.id, &status));

        return {status};
    }

    template <typename T>
    inline Status receive
        (   T& buffer
        ,   const Node& source = any
        ,   const Tag& tag = MPI_TAG_UB
        ,   const Communicator& comm = world    )
    {
        detail::Payload<T> payload {buffer};

        using P = typename decltype(payload)::type;

        int size = probe(source, tag, comm).getCount<P>();
        return receive(payload.resize(size), size, source, tag, comm);
    }
    /**#@-*/

    /**#@+
     * Gathers data from all nodes and deliver combined data to all nodes
     * @tparam T The type of buffer data to gather.
     * @tparam U The type of buffer data to gather.
     * @param send The outgoing buffer.
     * @param recv The incoming buffer.
     * @param scount The outgoing buffer size.
     * @param rcount The size of incoming buffer from each node.
     * @param displ The data displacement of each node.
     * @param comm The communicator this operation applies to.
     */
    template <typename T>
    inline void allgather
        (   T *send, int scount
        ,   T *recv, int rcount
        ,   const Communicator& comm = world    )
    {
        call(MPI_Allgather(send, scount, datatype::get<T>(), recv, rcount, datatype::get<T>(), comm.id));
    }

    template <typename T>
    inline void allgather
        (   T *send, int scount
        ,   T *recv, int *rcount, int *displ
        ,   const Communicator& comm = world    )
    {
        call(MPI_Allgatherv(send, scount, datatype::get<T>(), recv, rcount, displ, datatype::get<T>(), comm.id));
    }

    template <typename T, typename U>
    inline void allgather
        (   T& send, int scount, int displ
        ,   U& recv
        ,   const Communicator& comm = world    )
    {
        detail::Payload<T> sendl {send};
        detail::Payload<U> recvl {recv};

        using S = typename decltype(sendl)::type;
        using R = typename decltype(recvl)::type;

        static_assert(std::is_same<S, R>::value, "Cannot gather with different types!");

        std::vector<int> sizeList(comm.size), displList(comm.size);

        allgather(&scount, 1, sizeList.data(), 1, comm);
        allgather(&displ, 1, displList.data(), 1, comm);

        recvl.resize(std::accumulate(sizeList.begin(), sizeList.end(), 0));
        allgather(sendl.getBuffer(), sendl.getSize(), recvl.getBuffer(), sizeList.data(), displList.data(), comm);
    }

    template <typename T, typename U>
    inline void allgather
        (   T& send
        ,   U& recv
        ,   const Communicator& comm = world    )
    {
        detail::Payload<T> sendl {send};
        detail::Payload<U> recvl {recv};

        using S = typename decltype(sendl)::type;
        using R = typename decltype(recvl)::type;

        static_assert(std::is_same<S, R>::value, "Cannot gather with different types!");

        int size = sendl.getSize();
        std::vector<int> sizeList(comm.size), displList(comm.size + 1);

        allgather(&size, 1, sizeList.data(), 1, comm);

        bool equal = std::all_of(sizeList.begin(), sizeList.end(), [&size](int i) { return i == size; });
        std::partial_sum(sizeList.begin(), sizeList.end(), displList.begin() + 1);

        recvl.resize(displList.back());

        if(equal) allgather(sendl.getBuffer(), size, recvl.getBuffer(), size, comm);
        else allgather(sendl.getBuffer(), size, recvl.getBuffer(), sizeList.data(), displList.data(), comm);
    }
    /**#@-*/

    /**#@+
     * Gather data from nodes according to given distribution.
     * @tparam T The type of buffer data to gather.
     * @tparam U The type of buffer data to gather.
     * @param send The outgoing buffer.
     * @param recv The incoming buffer.
     * @param scount The outgoing buffer size.
     * @param rcount The size of incoming buffer from each node.
     * @param displ The data displacement of each node.
     * @param root The operation's root node.
     * @param comm The communicator this operation applies to.
     */
    template <typename T>
    inline void gather
        (   T *send, int scount
        ,   T *recv, int rcount
        ,   const Node& root = node::master
        ,   const Communicator& comm = world    )
    {
        call(MPI_Gather(send, scount, datatype::get<T>(), recv, rcount, datatype::get<T>(), root, comm.id));
    }

    template <typename T>
    inline void gather
        (   T *send, int scount
        ,   T *recv, int *rcount, int *displ
        ,   const Node& root = node::master
        ,   const Communicator& comm = world            )
    {
        call(MPI_Gatherv(send, scount, datatype::get<T>(), recv, rcount, displ, datatype::get<T>(), root, comm.id));
    }

    template <typename T, typename U>
    inline void gather
        (   T& send, int scount, int displ
        ,   U& recv
        ,   const Node& root = node::master
        ,   const Communicator& comm = world    )
    {
        detail::Payload<T> sendl {send};
        detail::Payload<U> recvl {recv};

        using S = typename decltype(sendl)::type;
        using R = typename decltype(recvl)::type;

        static_assert(std::is_same<S, R>::value, "Cannot gather with different types!");

        std::vector<int> sizeList(comm.size), displList(comm.size);

        gather(&scount, 1, sizeList.data(), 1, root, comm);
        gather(&displ, 1, displList.data(), 1, root, comm);

        if(comm.rank == root) recvl.resize(std::accumulate(sizeList.begin(), sizeList.end(), 0));
        gather(sendl.getBuffer(), sendl.getSize(), recvl.getBuffer(), sizeList.data(), displList.data(), root, comm);
    }

    template <typename T, typename U>
    inline void gather
        (   T& send
        ,   U& recv
        ,   const Node& root = node::master
        ,   const Communicator& comm = world    )
    {
        detail::Payload<T> sendl {send};
        detail::Payload<U> recvl {recv};

        using S = typename decltype(sendl)::type;
        using R = typename decltype(recvl)::type;

        static_assert(std::is_same<S, R>::value, "Cannot gather with different types!");

        int size = sendl.getSize();
        std::vector<int> sizeList(comm.size), displList(comm.size + 1);

        allgather(&size, 1, sizeList.data(), 1, comm);

        bool equal = std::all_of(sizeList.begin(), sizeList.end(), [&size](int i) { return i == size; });
        std::partial_sum(sizeList.begin(), sizeList.end(), displList.begin() + 1);

        if(comm.rank == root) recvl.resize(displList.back());

        if(equal) gather(sendl.getBuffer(), size, recvl.getBuffer(), size, root, comm);
        else gather(sendl.getBuffer(), size, recvl.getBuffer(), sizeList.data(), displList.data(), root, comm);
    }
    /**#@-*/

    /**#@+
     * Scatters data to nodes according to given distribution.
     * @tparam T The type of buffer data to scatter.
     * @tparam U The type of buffer data to gather.
     * @param send The outgoing buffer.
     * @param recv The incoming buffer.
     * @param scount The outgoing buffer size.
     * @param rcount The size of incoming buffer from each node.
     * @param displ The data displacement of each node.
     * @param root The operation's root node.
     * @param comm The communicator this operation applies to.
     */
    template <typename T>
    inline void scatter
        (   T *send, int scount
        ,   T *recv, int rcount
        ,   const Node& root = node::master
        ,   const Communicator& comm = world    )
    {
        call(MPI_Scatter(send, scount, datatype::get<T>(), recv, rcount, datatype::get<T>(), root, comm.id));
    }

    template <typename T>
    inline void scatter
        (   T *send, int *scount, int *displ
        ,   T *recv, int rcount
        ,   const Node& root = node::master
        ,   const Communicator& comm = world    )
    {
        call(MPI_Scatterv(send, scount, displ, datatype::get<T>(), recv, rcount, datatype::get<T>(), root, comm.id));
    }

    template <typename T, typename U>
    inline void scatter
        (   T& send
        ,   U& recv, int rcount, int displ
        ,   const Node& root = node::master
        ,   const Communicator& comm = world    )
    {
        detail::Payload<T> sendl {send};
        detail::Payload<U> recvl {recv};

        using S = typename decltype(sendl)::type;
        using R = typename decltype(recvl)::type;

        static_assert(std::is_same<S, R>::value, "Cannot scatter with different types!");

        std::vector<int> sizeList, displList;

        gather(rcount, sizeList, root, comm);
        gather(displ, displList, root, comm);

        scatter(sendl.getBuffer(), sizeList.data(), displList.data(), recvl.resize(rcount), rcount, root, comm);
    }

    template <typename T, typename U>
    inline void scatter
        (   T& send
        ,   U& recv
        ,   const Node& root = node::master
        ,   const Communicator& comm = world    )
    {
        detail::Payload<T> sendl {send};
        detail::Payload<U> recvl {recv};

        using S = typename decltype(sendl)::type;
        using R = typename decltype(recvl)::type;

        static_assert(std::is_same<S, R>::value, "Cannot gather with different types!");

        int size = sendl.getSize();
        broadcast(size, root, comm);

        int quo = size / comm.size;
        int rem = size % comm.size;

        recvl.resize(size = quo + (rem > comm.rank));

        if(!rem) scatter(sendl.getBuffer(), size, recvl.getBuffer(), size, root, comm);
        else scatter(send, recv, size, quo * comm.rank + std::min(comm.rank, rem), root, comm);
    }
    /**#@-*/

    /**
     * Synchronizes all nodes in a communicator.
     * @param comm The communicator the operation applies to.
     */
    inline void barrier(const Communicator& comm = world)
    {
        call(MPI_Barrier(comm.id));
    }

    /**
     * Finalizes all cluster communication operations between nodes.
     * @see mpi::init
     */
    inline void finalize()
    {
        for(MPI_Datatype& dtype : datatype::dtypes)
            call(MPI_Type_free(&dtype));
        call(MPI_Finalize());
    }
};

/**
 * Checks whether two communicators are the same.
 * @param a The first communicator to compare.
 * @param b The second communicator to compare.
 * @return Are both communicators the same?
 */
inline bool operator==(const mpi::Communicator& a, const mpi::Communicator& b)
{
    return a.id == b.id;
}

/**
 * Checks whether two communicators are different.
 * @param a The first communicator to compare.
 * @param b The second communicator to compare.
 * @return Are both communicators different?
 */
inline bool operator!=(const mpi::Communicator& a, const mpi::Communicator& b)
{
    return a.id != b.id;
}

#endif
#endif