/** 
 * Multiple Sequence Alignment MPI wrapper header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef MPI_WRAPPER_INCLUDED
#define MPI_WRAPPER_INCLUDED

#include <utils.hpp>

#if !defined(onlycython)

#include <map>
#include <mpi.h>
#include <string>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <utility>
#include <algorithm>

#include <node.hpp>
#include <tuple.hpp>
#include <buffer.hpp>
#include <exception.hpp>
#include <reflection.hpp>

namespace mpi
{
    /**
     * Represents the identifier of a node connected to the cluster.
     * @since 0.1.1
     */
    using node = ::node::id;

    /**
     * The value acceptable as a node source and a tag for a received message.
     * @since 0.1.1
     */
    enum : int32_t { any = -1 };

    /**
     * Indicates an API error or represents any error during execution.
     * @since 0.1.1
     */
    using error_code = int;

    namespace error
    {
        /**
         * Aliases for MPI error types and error codes enumeration.
         * @since 0.1.1
         */
        enum : error_code
        {
            success                 = MPI_SUCCESS
        ,   access                  = MPI_ERR_ACCESS
        ,   amode                   = MPI_ERR_AMODE
        ,   arg                     = MPI_ERR_ARG
        ,   assert                  = MPI_ERR_ASSERT
        ,   bad_file                = MPI_ERR_BAD_FILE
        ,   base                    = MPI_ERR_BASE
        ,   buffer                  = MPI_ERR_BUFFER
        ,   comm                    = MPI_ERR_COMM
        ,   conversion              = MPI_ERR_CONVERSION
        ,   count                   = MPI_ERR_COUNT
        ,   dims                    = MPI_ERR_DIMS
        ,   disp                    = MPI_ERR_DISP
        ,   dup_datarep             = MPI_ERR_DUP_DATAREP
        ,   file                    = MPI_ERR_FILE
        ,   file_exists             = MPI_ERR_FILE_EXISTS
        ,   file_in_use             = MPI_ERR_FILE_IN_USE
        ,   group                   = MPI_ERR_GROUP
        ,   in_status               = MPI_ERR_IN_STATUS
        ,   info                    = MPI_ERR_INFO
        ,   info_key                = MPI_ERR_INFO_KEY
        ,   info_nokey              = MPI_ERR_INFO_NOKEY
        ,   info_value              = MPI_ERR_INFO_VALUE
        ,   intern                  = MPI_ERR_INTERN
        ,   io                      = MPI_ERR_IO
        ,   keyval                  = MPI_ERR_KEYVAL
        ,   lastcode                = MPI_ERR_LASTCODE
        ,   locktype                = MPI_ERR_LOCKTYPE
        ,   name                    = MPI_ERR_NAME
        ,   no_mem                  = MPI_ERR_NO_MEM
        ,   no_space                = MPI_ERR_NO_SPACE
        ,   no_such_file            = MPI_ERR_NO_SUCH_FILE
        ,   not_same                = MPI_ERR_NOT_SAME
        ,   op                      = MPI_ERR_OP
        ,   other                   = MPI_ERR_OTHER
        ,   pending                 = MPI_ERR_PENDING
        ,   port                    = MPI_ERR_PORT
        ,   quota                   = MPI_ERR_QUOTA
        ,   rank                    = MPI_ERR_RANK
        ,   read_only               = MPI_ERR_READ_ONLY
        ,   request                 = MPI_ERR_REQUEST
        ,   rma_conflict            = MPI_ERR_RMA_CONFLICT
        ,   rma_sync                = MPI_ERR_RMA_SYNC
        ,   root                    = MPI_ERR_ROOT
        ,   service                 = MPI_ERR_SERVICE
        ,   size                    = MPI_ERR_SIZE
        ,   spawn                   = MPI_ERR_SPAWN
        ,   tag                     = MPI_ERR_TAG
        ,   topology                = MPI_ERR_TOPOLOGY
        ,   truncate                = MPI_ERR_TRUNCATE
        ,   type                    = MPI_ERR_TYPE
        ,   unsupported_datarep     = MPI_ERR_UNSUPPORTED_DATAREP
        ,   unsupported_operation   = MPI_ERR_UNSUPPORTED_OPERATION
        ,   win                     = MPI_ERR_WIN
        ,   unknown                 = MPI_ERR_UNKNOWN
        };

        /**
         * Produces an error message explaining error obtained by code.
         * @param code The error code to be explained.
         * @return The error description.
         */
        inline std::string describe(error_code code) noexcept
        {
            int ignore;
            char msgbuf[MPI_MAX_ERROR_STRING];

            return MPI_Error_string(code, msgbuf, &ignore) != success
                ? "error while probing MPI error"
                : msgbuf;
        }
    }

    /**
     * Holds a MPI error message so it can be propagated through the code.
     * @since 0.1.1
     */
    class exception : public ::exception
    {
        protected:
            using underlying_type = ::exception;    /// The underlying exception.

        protected:
            error_code mcode;                       /// The status code.

        public:     
            /**
             * Builds a new exception instance.
             * @param code The result code reported by MPI.
             */
            inline exception(error_code code) noexcept
            :   underlying_type {"mpi exception: %s", error::describe(code)}
            ,   mcode {code}
            {}

            /**
             * Builds a new exception instance from error code.
             * @tparam T The format parameters' types.
             * @param code The error code.
             * @param fmtstr The additional message's format.
             * @param args The format's parameters.
             */
            template <typename ...T>
            inline exception(error_code code, const std::string& fmtstr, T&&... args) noexcept
            :   underlying_type {fmtstr, args...}
            ,   mcode {code}
            {}

            using underlying_type::exception;

            /**
             * Retrieves the MPI error code thrown.
             * @param The error code.
             */
            inline error_code code() const noexcept
            {
                return mcode;
            }
    };

    /**
     * Checks whether a MPI operation has been successful and throws error if not.
     * @param code The error code obtained from the operation.
     * @throw The error code obtained raised to exception.
     */
    inline void check(error_code code)
    {
        enforce<exception>(code == error::success, code);
    }

    namespace datatype
    {
        /**
         * Holds an identification value for a type that can be sent via MPI.
         * @since 0.1.1
         */
        using id = MPI_Datatype;

        /*
         * Forward-declaration of type adapter. This class is responsible for
         * automatically adapting a new type for MPI.
         */
        template <typename T>
        class adapter;

        /**
         * Gives access to the raw datatype for the requested type.
         * @tparam T The requested type.
         * @return The datatype created for type T.
         */
        template <typename T> inline id get() { return adapter<T>::get(); }

        /**#@+
         * Template specializations for built-in types. These types are created
         * for built-in types automatically and, thus, can be used directly.
         * @since 0.1.1
         */
        template <> inline id get<bool>()     { return MPI_C_BOOL; };
        template <> inline id get<char>()     { return MPI_CHAR; };
        template <> inline id get<float>()    { return MPI_FLOAT; };
        template <> inline id get<double>()   { return MPI_DOUBLE; };
        template <> inline id get<int8_t>()   { return MPI_INT8_T; };
        template <> inline id get<uint8_t>()  { return MPI_UINT8_T; };
        template <> inline id get<int16_t>()  { return MPI_INT16_T; };
        template <> inline id get<uint16_t>() { return MPI_UINT16_T; };
        template <> inline id get<int32_t>()  { return MPI_INT32_T; };
        template <> inline id get<uint32_t>() { return MPI_UINT32_T; };
        template <> inline id get<int64_t>()  { return MPI_INT64_T; };
        template <> inline id get<uint64_t>() { return MPI_UINT64_T; };
        /**#@-*/
    }
}

namespace internal
{
    namespace mpi
    {
        /**
         * The initial step for generating new datatypes.
         * @tparam T The type to be represented by the datatype.
         * @since 0.1.1
         */
        template <typename T>
        struct type_builder
        {
            /**
             * Initializes the recursion through the type's reflection tuple.
             * @param blocks The list of block lengths in type T.
             * @param offsets The list of field offsets in type T.
             * @param types The list of datatypes in type T.
             */
            inline static void generate(int *blocks, MPI_Aint *offsets, MPI_Datatype *types)
            {
                type_builder<::reflection_tuple<T>>::template generate<T>(blocks, offsets, types);
            }
        };

        /**
         * The final recursion step for creating a new datatype.
         * @param T The last type in reflection tuple to become part of the datatype.
         * @since 0.1.1
         */
        template <typename T>
        struct type_builder<::tuple<T>>
        {
            /**
             * Terminates the recursion through the original type's reflection tuple.
             * @tparam O The original type corresponding to the datatype.
             * @tparam N The current recurstion step.
             * @param blocks The list of block lengths in type T.
             * @param offsets The list of field offsets in type T.
             * @param types The list of datatypes in type T.
             */
            template <typename O, size_t N = 0>
            inline static void generate(int *blocks, MPI_Aint *offsets, MPI_Datatype *types)
            {
                blocks[N] = utils::max(int(std::extent<T>::value), 1);
                offsets[N] = ::reflection<O>::template offset<N>();
                types[N] = ::mpi::datatype::get<base<T>>();
            }
        };

        /**
         * The middle recursion steps for creating a new datatype.
         * @tparam T The current type in reflection tuple to become part of the datatype.
         * @tparam U The following types in reflection tuple.
         * @since 0.1.1
         */
        template <typename T, typename ...U>
        struct type_builder<::tuple<T, U...>>
        {
            /**
             * Processes a step of the recursion through the original type's reflection tuple.
             * @tparam O The original type corresponding to the datatype.
             * @tparam N The current recurstion step.
             * @param blocks The list of block lengths in type T.
             * @param offsets The list of field offsets in type T.
             * @param types The list of datatypes in type T.             
             */
            template <typename O, size_t N = 0>
            inline static void generate(int *blocks, MPI_Aint *offsets, MPI_Datatype *types)
            {
                type_builder<::tuple<T>>::template generate<O, N>(blocks, offsets, types);
                type_builder<::tuple<U...>>::template generate<O, N + 1>(blocks, offsets, types);
            }
        };
    }
}

namespace mpi
{
    namespace datatype
    {
        /**
         * Keeps track of all generated datatypes throughout execution.
         * @since 0.1.1
         */
        extern std::vector<id> ref_type;

        /**
         * Adapts a new datatype for a user defined type.
         * @tparam T The type to which datatype must be created.
         * @since 0.1.1
         */
        template <typename T>
        class adapter
        {
            static_assert(!std::is_union<T>::value, "unions should not be sent via mpi");

            private:
                id mtypeid;             /// The raw MPI datatype reference.

            public:
                /**
                 * Builds a new instance and creates the datatype for the requested
                 * type. This constructor shall be called only once for each type
                 * to be created.
                 * @see adapter::get
                 */
                inline adapter()
                {
                    constexpr size_t count = reflection<T>::count();

                    int blocks[count];
                    MPI_Aint offsets[count];
                    MPI_Datatype types[count];

                    internal::mpi::type_builder<T>::generate(blocks, offsets, types);

                    check(MPI_Type_create_struct(count, blocks, offsets, types, &mtypeid));
                    check(MPI_Type_commit(&mtypeid));
                    ref_type.push_back(mtypeid);
                }

                /**
                 * Gives access to the datatype created for the requested type.
                 * @return The datatype created for type T.
                 */
                inline static id get()
                {
                    static adapter instance;
                    return instance.mtypeid;
                }
        };
    }

    /**
     * Contains information about a message that has been or can be received.
     * @since 0.1.1
     */
    class status
    {
        public:
            using raw_type = MPI_Status;    /// The raw MPI status type.

        protected:
            mutable raw_type mraw;          /// The raw status info.

        public:
            inline status() noexcept = default;
            inline status(const status&) noexcept = default;

            /**
             * Instatiates a new status object.
             * @param builtin The MPI status built-in object.
             */
            inline status(const raw_type& builtin) noexcept
            :   mraw {builtin}
            {}

            /**
             * Converts to the built-in status object.
             * @return The built-in status object.
             */
            inline operator const raw_type&() const noexcept
            {
                return mraw;
            }

            /**
             * Retrieves the message error code.
             * @return The error code.
             */
            inline error_code error() const noexcept
            {
                return mraw.MPI_ERROR;
            }

            /**
             * Retrieves the source of the message.
             * @return The message source node.
             */
            inline node source() const noexcept
            {
                return mraw.MPI_SOURCE;
            }

            /**
             * Retrieves the message tag.
             * @return The retrieved message tag.
             */
            inline int32_t tag() const noexcept
            {
                return mraw.MPI_TAG;
            }

            /**
             * Determines the number of elements contained in the message.
             * @tparam T The message content type.
             * @return The number of elements contained in the message.
             */
            template <typename T>
            inline int32_t count() const noexcept
            {
                int value;
                MPI_Get_count(&mraw, datatype::get<T>(), &value);
                return (value != MPI_UNDEFINED) ? value : -1;
            }

            /**
             * Determines whether the communication associated with this object
             * has been successfully cancelled.
             * @return Has the message been cancelled?
             */
            inline bool cancelled() const noexcept
            {
                int flag = 0;
                MPI_Test_cancelled(&mraw, &flag);
                return (flag != 0);
            }
    };
}

namespace internal
{
    namespace mpi
    {
        /**
         * Represents incoming and outcoming message payload of communication operations.
         * In practice, this object serves as a neutral context state for messages.
         * @tparam T The message payload type.
         * @since 0.1.1
         */
        template <typename T>
        class payload
        {
            public:
                using element_type = pure<T>;       /// The payload's elementary type.

            protected:
                element_type *mptr = nullptr;       /// The payload's buffer pointer.
                size_t msize = 0;                   /// The payload's number of elements.

            public:
                inline payload() noexcept = delete;
                inline payload(const payload&) noexcept = default;
                inline payload(payload&&) noexcept = default;

                /**
                 * Creates a new payload from simple object value.
                 * @param value The payload's value.
                 */
                inline payload(element_type& value) noexcept
                :   mptr {&value}
                ,   msize {1}
                {}

                /**
                 * Creates a new payload from already existing buffer.
                 * @param ptr The payload's buffer pointer.
                 * @param size The payload's buffer size.
                 */
                inline payload(element_type *ptr, size_t size = 1) noexcept
                :   mptr {ptr}
                ,   msize {size}
                {}

                /**
                 * Handles any memory transfers, swaps or deallocations needed
                 * for an effective clean-up of this object.
                 */
                inline virtual ~payload() noexcept = default;

                inline payload& operator=(const payload&) noexcept = default;
                inline payload& operator=(payload&&) noexcept = default;

                /**
                 * Retrieves the payload's buffer pointer.
                 * @return The payload's buffer pointer.
                 */
                inline element_type *data() const noexcept
                {
                    return mptr;
                }

                /**
                 * Retrieves the payload's buffer capacity.
                 * @return The payload's size or capacity.
                 */
                inline size_t size() const noexcept
                {
                    return msize;
                }

                /**
                 * Creates a new pointer with given size and swaps buffers. This allow
                 * the payload to receive an incoming message.
                 * @param (ignored) The new minimum payload size.
                 * @return The resized buffer pointer.
                 */
                inline virtual element_type *resize(size_t)
                {
                    return data();
                }
        };

        /**
         * Message payload context for STL vectors.
         * @tparam T The message payload type.
         * @since 0.1.1
         */
        template <typename T>
        class payload<std::vector<T>> : public payload<T>
        {
            public:
                using element_type = pure<T>;       /// The payload's elementary type.

            protected:
                std::vector<T>& mref;               /// The original vector's reference.

            public:
                /**
                 * Creates a new payload from STL vector.
                 * @param ref The payload as a STL vector.
                 */
                inline payload(std::vector<T>& ref) noexcept
                :   payload<T> {ref.data(), ref.size()}
                ,   mref {ref}
                {}

                /**
                 * Creates a new vector and swaps buffers.
                 * @param size The new minimum payload capacity.
                 * @return The resized buffer pointer.
                 */
                inline element_type *resize(size_t size) override
                {
                    if(this->msize < size)
                        mref.resize(this->msize = size);
                    return this->mptr = mref.data();
                }
        };

        /**
         * Message payload context for buffers.
         * @tparam T The message payload type.
         * @since 0.1.1
         */
        template <typename T>
        class payload<buffer<T>> : public payload<T>
        {
            public:
                using element_type = pure<T>;       /// The payload's elementary type.

            protected:
                buffer<T>& mref;                    /// The original buffer's reference.

            public:
                /**
                 * Creates a new payload from buffer.
                 * @param ref The buffer to use as payload.
                 */
                inline payload(buffer<T>& ref) noexcept
                :   payload<T> {ref.raw(), ref.size()}
                ,   mref {ref}
                {}

                /**
                 * Creates a new buffer object and swaps contents.
                 * @param size The new minimum payload capacity.
                 * @return The new buffer pointer
                 */
                inline element_type *resize(size_t size) override
                {
                    if(this->msize != size)
                        mref = buffer<T>::make(mref.allocator(), this->msize = size);
                    return this->mptr = mref.raw();
                }
        };
    }
}

namespace mpi
{
    namespace op
    {
        /**
         * Identifies an operator for MPI collective operations.
         * @since 0.1.1
         */
        using id = MPI_Op;

        /**
         * Keeps track of all user defined operator's ids created during execution.
         * @since 0.1.1
         */
        extern std::vector<id> ref_op;

        /**
         * Maps a datatype to an user-created operator. This is necessary because
         * it is almost technically impossible to inject the operator inside the
         * wrapper without an extremelly convoluted mechanism.
         * @since 0.1.1
         */
        extern std::map<id, void *> op_list;

        /**
         * Informs the currently active operator. This will be useful for injecting
         * the correct operator inside the wrapper.
         * @since 0.1.1
         */
        extern id active;

        /**
         * Wraps an operator transforming it into an MPI operator.
         * @tparam T The type the operator works onto.
         * @param a The operation's first operand.
         * @param b The operation's second operand and output value.
         * @param len The number of elements in given operation.
         */
        template <typename T>
        void fwrap(const void *a, void *b, int *len, MPI_Datatype *)
        {
            using function_type = typename utils::op<T>::function_type;
            auto f = reinterpret_cast<function_type>(op_list[active]);

            for(int i = 0; i < *len; ++i)
                static_cast<T*>(b)[i] = f(static_cast<T*>(a)[i], static_cast<T*>(b)[i]);
        }

        /**
         * Creates a new MPI operator from user function.
         * @tparam T The type the operator works onto.
         * @param func The functor of operator to be created.
         * @param commutative Is the operator commutative?
         * @return The identifier for operator created.
         */
        template <typename T>
        inline id create(utils::op<T> func, bool commutative = true)
        {
            id result;
            check(MPI_Op_create(fwrap<T>, commutative, &result));
            op_list[result] = reinterpret_cast<void *>(&func);
            ref_op.push_back(result);
            return result;
        }

        /**#@+
         * Listing of built-in operators. The use of these instead of creating
         * new operators is highly recommended.
         * @since 0.1.1
         */
        static constexpr id const& max      = MPI_MAX;
        static constexpr id const& min      = MPI_MIN;
        static constexpr id const& sum      = MPI_SUM;
        static constexpr id const& mul      = MPI_PROD;
        static constexpr id const& andl     = MPI_LAND;
        static constexpr id const& andb     = MPI_BAND;
        static constexpr id const& orl      = MPI_LOR;
        static constexpr id const& orb      = MPI_BOR;
        static constexpr id const& xorl     = MPI_LXOR;
        static constexpr id const& xorb     = MPI_BXOR;
        static constexpr id const& minloc   = MPI_MINLOC;
        static constexpr id const& maxloc   = MPI_MAXLOC;
        static constexpr id const& replace  = MPI_REPLACE;
        /**#@-*/
    }

    namespace communicator
    {
        /**
         * Permits communication and synchronization among a set of nodes and processes.
         * @since 0.1.1
         */
        struct id
        {
            node rank = 0;                  /// The rank of current node in relation to communicator.
            uint32_t size = 0;              /// The number of nodes in the communicator.
            MPI_Comm ref = MPI_COMM_NULL;   /// The communicator internal reference.
        };

        /**
         * Represents a false, non-existent or invalid communicator.
         * @since 0.1.1
         */
        static constexpr id null;

        /**
         * Builds up a new communicator instance from built-in type.
         * @param comm Built-in communicator instance.
         * @return The new communicator instance.
         */
        inline id build(MPI_Comm comm)
        {
            int rank, size;
            check(MPI_Comm_rank(comm, &rank));
            check(MPI_Comm_size(comm, &size));
            return {rank, static_cast<uint32_t>(size), comm};
        }

        /**
         * Clones the communicator, creating a new communicator as a copy.
         * @param comm The communicator to clone.
         * @return The clone created from original communicator.
         */
        inline id clone(const id& comm)
        {
            MPI_Comm newcomm;
            check(MPI_Comm_dup(comm.ref, &newcomm));
            check(MPI_Comm_set_errhandler(newcomm, MPI_ERRORS_RETURN));
            return build(newcomm);
        }

        /**
         * Splits nodes into different communicators according to selected color.
         * @param comm The original communicator to be split.
         * @param color The color selected by current node.
         * @param key The key used to assigned a node id in new communicator.
         * @return The obtained communicator from split operation.
         */
        inline id split(const id& comm, int color, int key = -1)
        {
            MPI_Comm newcomm;
            check(MPI_Comm_split(comm.ref, color, (key > 0 ? key : comm.rank), &newcomm));
            return build(newcomm);
        }

        /**
         * Cleans up resources used by communicator.
         * @param comm The communicator to be destroyed.
         */
        inline void free(id& comm)
        {
            check(MPI_Comm_free(&comm.ref));
            comm = null;
        }
    }

    /**
     * The default communicator instance.
     * @see mpi::communicator::id
     */
    extern communicator::id world;

    /*
     * Global MPI initialization and finalization routines.
     */
    void init(int&, char **&);
    void finalize();

    /**
     * Creates a new payload from given buffer.
     * @tparam T The given buffer content type.
     * @param tgt The base payload buffer.
     * @return The new payload.
     */
    template <typename T>
    inline auto payload(T& tgt) noexcept -> internal::mpi::payload<T>
    {
        return {tgt};
    }

    /**#@+
     * Broadcasts data to all nodes in given communicator.
     * @tparam T Type of buffer data to broadcast.
     * @param tgt The target buffer to broadcast.
     * @param count The number of buffer's elements to broadcast.
     * @param comm The communicator this operation applies to.
     * @param root The operation's root node.
     */
    template <typename T>
    inline void broadcast(
            T *tgt
        ,   int count = 1
        ,   const node& root = ::node::master
        ,   const communicator::id& comm = world
        )
    {
        check(MPI_Bcast(tgt, count, datatype::get<T>(), root, comm.ref));
    }

    template <typename T>
    inline void broadcast(
            T& tgt
        ,   const node& root = ::node::master
        ,   const communicator::id& comm = world
        )
    {
        auto pload = payload(tgt);
        auto size = pload.size();

        broadcast(&size, 1, root, comm);
        broadcast(pload.resize(size), size, root, comm);
    }
    /**#@-*/

    /**
     * Inspects incoming message and retrieves its status.
     * @param src The source node.
     * @param t The identifying message tag.
     * @param comm The communicator this operation applies to.
     * @return The inspected message status.
     */
    inline status probe(
            const node& src = ::node::master
        ,   const int32_t& tag = any
        ,   const communicator::id& comm = world
        )
    {
        MPI_Status stt;
        check(MPI_Probe(src, tag < 0 ? MPI_TAG_UB : tag, comm.ref, &stt));
        return {stt};
    }

    /**#@+
     * Sends data to a node connected to the cluster.
     * @tparam T Type of buffer data to send.
     * @param tgt The buffer to send.
     * @param count The number of buffer's elements to send.
     * @param dest The destination node.
     * @param t The identifying message tag.
     * @param comm The communicator this operation applies to.
     * @return MPI error code if not successful.
     */
    template <typename T>
    inline void send(
            T *tgt
        ,   int count = 1
        ,   const node& dest = ::node::master
        ,   const int32_t& tag = MPI_TAG_UB
        ,   const communicator::id& comm = world
        )
    {
        check(MPI_Send(tgt, count, datatype::get<T>(), dest, tag < 0 ? MPI_TAG_UB : tag, comm.ref));
    }

    template <typename T>
    inline void send(
            T& tgt
        ,   const node& dest = ::node::master
        ,   const int32_t& tag = MPI_TAG_UB
        ,   const communicator::id& comm = world
        )
    {
        auto pload = payload(tgt);
        send(pload.data(), pload.size(), dest, tag, comm);
    }
    /**#@-*/

    /**#@+
     * Receives data from a node connected to the cluster.
     * @tparam T Type of buffer data to receive.
     * @param tgt The buffer to receive data into.
     * @param count The number of buffer's elements to receive.
     * @param src The source node.
     * @param t The identifying tag.
     * @param comm The communicator this operation applies to.
     * @return The message status
     */
    template <typename T>
    inline status receive(
            T *tgt
        ,   int count = 1
        ,   const node& src = any
        ,   const int32_t& tag = MPI_TAG_UB
        ,   const communicator::id& comm = world
        )
    {
        MPI_Status stt;
        check(MPI_Recv(tgt, count, datatype::get<T>(), src, tag, comm.ref, &stt));
        return {stt};
    }

    template <typename T>
    inline status receive(
            T& tgt
        ,   const node& src = any
        ,   const int32_t& tag = MPI_TAG_UB
        ,   const communicator::id& comm = world
        )
    {
        auto pload = payload(tgt);
        using P = typename decltype(pload)::element_type;

        auto size = probe(src, tag, comm).count<P>();
        return receive(pload.resize(size), size, src, tag, comm);
    }
    /**#@-*/

    /**#@+
     * Gathers data from all nodes and deliver combined data to all nodes
     * @tparam T The type of buffer data to gather.
     * @tparam U The type of buffer data to gather.
     * @param out The outgoing buffer.
     * @param osz The outgoing buffer size.
     * @param in The incoming buffer.
     * @param isz The size of incoming buffer from each node.
     * @param displ The data displacement of each node.
     * @param comm The communicator this operation applies to.
     */
    template <typename T>
    inline void allgather(
            T *out, int osz
        ,   T *in, int isz
        ,   const communicator::id& comm = world
        )
    {
        check(MPI_Allgather(out, osz, datatype::get<T>(), in, isz, datatype::get<T>(), comm.ref));
    }

    template <typename T>
    inline void allgather(
            T *out, int osz
        ,   T *in, int *isz, int *displ
        ,   const communicator::id& comm = world
        )
    {
        check(MPI_Allgatherv(out, osz, datatype::get<T>(), in, isz, displ, datatype::get<T>(), comm.ref));
    }

    template <typename T, typename U>
    inline void allgather(
            T& out, int osz, int displ
        ,   U& in
        ,   const communicator::id& comm = world
        )
    {
        auto oload = payload(out);
        auto iload = payload(in);
        using O = typename decltype(oload)::element_type;
        using I = typename decltype(iload)::element_type;
        static_assert(std::is_same<O, I>::value, "cannot gather with different types");

        std::vector<int> lsize(comm.size), ldispl(comm.size);

        allgather(&osz, 1, lsize.data(), 1, comm);
        allgather(&displ, 1, ldispl.data(), 1, comm);

        iload.resize(std::accumulate(lsize.begin(), lsize.end(), 0));
        allgather(oload.data(), oload.size(), iload.data(), lsize.data(), ldispl.data(), comm);
    }

    template <typename T, typename U>
    inline void allgather(T& out, U& in, const communicator::id& comm = world)
    {
        auto oload = payload(out);
        auto iload = payload(in);
        using O = typename decltype(oload)::element_type;
        using I = typename decltype(iload)::element_type;
        static_assert(std::is_same<O, I>::value, "cannot gather with different types");

        int size = oload.size();
        std::vector<int> lsize(comm.size), ldispl(comm.size + 1);

        allgather(&size, 1, lsize.data(), 1, comm);

        bool equal = std::all_of(lsize.begin(), lsize.end(), [&size](int i) { return i == size; });
        std::partial_sum(lsize.begin(), lsize.end(), ldispl.begin() + 1);

        iload.resize(ldispl.back());

        if(equal) allgather(oload.data(), size, iload.data(), size, comm);
        else allgather(oload.data(), size, iload.data(), lsize.data(), ldispl.data(), comm);
    }
    /**#@-*/

    /**#@+
     * Gather data from nodes according to given distribution.
     * @tparam T The type of buffer data to gather.
     * @tparam U The type of buffer data to gather.
     * @param out The outgoing buffer.
     * @param in The incoming buffer.
     * @param osz The outgoing buffer size.
     * @param isz The size of incoming buffer from each node.
     * @param displ The data displacement of each node.
     * @param root The operation's root node.
     * @param comm The communicator this operation applies to.
     */
    template <typename T>
    inline void gather(
            T *out, int osz
        ,   T *in, int isz
        ,   const node& root = ::node::master
        ,   const communicator::id& comm = world
        )
    {
        check(MPI_Gather(out, osz, datatype::get<T>(), in, isz, datatype::get<T>(), root, comm.ref));
    }

    template <typename T>
    inline void gather(
            T *out, int osz
        ,   T *in, int *isz, int *displ
        ,   const node& root = ::node::master
        ,   const communicator::id& comm = world
        )
    {
        check(MPI_Gatherv(out, osz, datatype::get<T>(), in, isz, displ, datatype::get<T>(), root, comm.ref));
    }

    template <typename T, typename U>
    inline void gather(
            T& out, int osz, int displ
        ,   U& in
        ,   const node& root = ::node::master
        ,   const communicator::id& comm = world
        )
    {
        auto oload = payload(out);
        auto iload = payload(in);
        using O = typename decltype(oload)::element_type;
        using I = typename decltype(iload)::element_type;
        static_assert(std::is_same<O, I>::value, "cannot gather with different types");

        std::vector<int> lsize(comm.size), ldispl(comm.size);

        gather(&osz, 1, lsize.data(), 1, root, comm);
        gather(&displ, 1, ldispl.data(), 1, root, comm);

        if(comm.rank == root) iload.resize(std::accumulate(lsize.begin(), lsize.end(), 0));
        gather(oload.data(), oload.size(), iload.data(), lsize.data(), ldispl.data(), root, comm);
    }

    template <typename T, typename U>
    inline void gather(
            T& out
        ,   U& in
        ,   const node& root = ::node::master
        ,   const communicator::id& comm = world
        )
    {
        auto oload = payload(out);
        auto iload = payload(in);
        using O = typename decltype(oload)::element_type;
        using I = typename decltype(iload)::element_type;
        static_assert(std::is_same<O, I>::value, "cannot gather with different types");

        int size = oload.size();
        std::vector<int> lsize(comm.size), ldispl(comm.size + 1);

        allgather(&size, 1, lsize.data(), 1, comm);

        bool equal = std::all_of(lsize.begin(), lsize.end(), [&size](int i) { return i == size; });
        std::partial_sum(lsize.begin(), lsize.end(), ldispl.begin() + 1);

        if(comm.rank == root) iload.resize(ldispl.back());

        if(equal) gather(oload.data(), size, iload.data(), size, root, comm);
        else gather(oload.data(), size, iload.data(), lsize.data(), ldispl.data(), root, comm);
    }
    /**#@-*/

    /**#@+
     * Scatters data to nodes according to given distribution.
     * @tparam T The type of buffer data to scatter.
     * @tparam U The type of buffer data to gather.
     * @param out The outgoing buffer.
     * @param in The incoming buffer.
     * @param osz The outgoing buffer size.
     * @param isz The size of incoming buffer from each node.
     * @param displ The data displacement of each node.
     * @param root The operation's root node.
     * @param comm The communicator this operation applies to.
     */
    template <typename T>
    inline void scatter(
            T *out, int osz
        ,   T *in, int isz
        ,   const node& root = ::node::master
        ,   const communicator::id& comm = world
        )
    {
        check(MPI_Scatter(out, osz, datatype::get<T>(), in, isz, datatype::get<T>(), root, comm.ref));
    }

    template <typename T>
    inline void scatter(
            T *out, int *osz, int *displ
        ,   T *in, int isz
        ,   const node& root = ::node::master
        ,   const communicator::id& comm = world
        )
    {
        check(MPI_Scatterv(out, osz, displ, datatype::get<T>(), in, isz, datatype::get<T>(), root, comm.ref));
    }

    template <typename T, typename U>
    inline void scatter(
            T& out
        ,   U& in, int isz, int displ
        ,   const node& root = ::node::master
        ,   const communicator::id& comm = world
        )
    {
        auto oload = payload(out);
        auto iload = payload(in);
        using O = typename decltype(oload)::element_type;
        using I = typename decltype(iload)::element_type;
        static_assert(std::is_same<O, I>::value, "cannot scatter with different types");

        std::vector<int> lsize, ldispl;

        gather(isz, lsize, root, comm);
        gather(displ, ldispl, root, comm);

        scatter(oload.data(), lsize.data(), ldispl.data(), iload.resize(isz), isz, root, comm);
    }

    template <typename T, typename U>
    inline void scatter(
            T& out
        ,   U& in
        ,   const node& root = ::node::master
        ,   const communicator::id& comm = world
        )
    {
        auto oload = payload(out);
        auto iload = payload(in);
        using O = typename decltype(oload)::element_type;
        using I = typename decltype(iload)::element_type;
        static_assert(std::is_same<O, I>::value, "cannot scatter with different types");

        int size = oload.size();
        broadcast(size, root, comm);

        int quo = size / comm.size;
        int rem = size % comm.size;

        iload.resize(size = quo + (rem > comm.rank));

        if(!rem) scatter(oload.data(), size, iload.data(), size, root, comm);
        else scatter(out, in, size, quo * comm.rank + utils::min(comm.rank, rem), root, comm);
    }
    /**#@-*/

    /**
     * Synchronizes all nodes in a communicator.
     * @param comm The communicator the operation applies to.
     */
    inline void barrier(const communicator::id& comm = world)
    {
        check(MPI_Barrier(comm.ref));
    }
}

/**
 * Checks whether two communicators are the same.
 * @param a The first communicator to compare.
 * @param b The second communicator to compare.
 * @return Are both communicators the same?
 */
inline bool operator==(const mpi::communicator::id& a, const mpi::communicator::id& b) noexcept
{
    return a.ref == b.ref;
}

/**
 * Checks whether two communicators are different.
 * @param a The first communicator to compare.
 * @param b The second communicator to compare.
 * @return Are both communicators different?
 */
inline bool operator!=(const mpi::communicator::id& a, const mpi::communicator::id& b) noexcept
{
    return a.ref != b.ref;
}

#endif
#endif