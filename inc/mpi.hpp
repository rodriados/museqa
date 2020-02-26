/** 
 * Multiple Sequence Alignment MPI wrapper header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2020 Rodrigo Siqueira
 */
#pragma once

#include <environment.h>

#if !__msa(runtime, cython)

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
#include <utils.hpp>
#include <buffer.hpp>
#include <exception.hpp>
#include <reflection.hpp>

namespace msa
{
    namespace mpi
    {
        /**
         * Represents the identifier of a node connected to the cluster.
         * @since 0.1.1
         */
        using node = msa::node::id;

        /**
         * Represents a message transmission tag.
         * @since 0.1.1
         */
        using tag = int32_t;

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
                    ? "error while probing MPI error message"
                    : msgbuf;
            }
        }

        /**
         * Holds a MPI error message so it can be propagated through the code.
         * @since 0.1.1
         */
        class exception : public msa::exception
        {
            protected:
                using underlying_type = msa::exception; /// The underlying exception.

            protected:
                error_code m_code;                      /// The status code.

            public:     
                /**
                 * Builds a new exception instance.
                 * @param code The result code reported by MPI.
                 */
                inline exception(error_code code) noexcept
                :   underlying_type {"mpi exception: %s", error::describe(code)}
                ,   m_code {code}
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
                ,   m_code {code}
                {}

                using underlying_type::exception;

                /**
                 * Retrieves the MPI error code thrown.
                 * @param The error code.
                 */
                inline error_code code() const noexcept
                {
                    return m_code;
                }
        };

        /**
         * Checks whether a MPI operation has been successful and throws exception.
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

    namespace detail
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
                    type_builder<msa::reflection_tuple<T>>::template generate<T>(blocks, offsets, types);
                }
            };

            /**
             * The final recursion step for creating a new datatype.
             * @param T The last type in reflection tuple to become part of the datatype.
             * @since 0.1.1
             */
            template <typename T>
            struct type_builder<msa::tuple<T>>
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
                    blocks[N] = msa::utils::max(int(std::extent<T>::value), 1);
                    offsets[N] = msa::reflection<O>::template offset<N>();
                    types[N] = msa::mpi::datatype::get<msa::base<T>>();
                }
            };

            /**
             * The middle recursion steps for creating a new datatype.
             * @tparam T The current type in reflection tuple to become part of the datatype.
             * @tparam U The following types in reflection tuple.
             * @since 0.1.1
             */
            template <typename T, typename ...U>
            struct type_builder<msa::tuple<T, U...>>
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
                    type_builder<msa::tuple<T>>::template generate<O, N>(blocks, offsets, types);
                    type_builder<msa::tuple<U...>>::template generate<O, N + 1>(blocks, offsets, types);
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
                    id m_typeid;             /// The raw MPI datatype reference.

                public:
                    /**
                     * Builds a new instance and creates the datatype for the requested
                     * type. This constructor shall be called only once for each
                     * type to be created.
                     * @see adapter::get
                     */
                    inline adapter()
                    {
                        constexpr size_t count = reflection<T>::count();

                        int blocks[count];
                        MPI_Aint offsets[count];
                        MPI_Datatype types[count];

                        detail::mpi::type_builder<T>::generate(blocks, offsets, types);

                        check(MPI_Type_create_struct(count, blocks, offsets, types, &m_typeid));
                        check(MPI_Type_commit(&m_typeid));
                        ref_type.push_back(m_typeid);
                    }

                    /**
                     * Gives access to the datatype created for the requested type.
                     * @return The datatype created for type T.
                     */
                    inline static id get()
                    {
                        static adapter instance;
                        return instance.m_typeid;
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
                mutable raw_type m_raw;          /// The raw status info.

            public:
                inline status() noexcept = default;
                inline status(const status&) noexcept = default;
                inline status(status&&) noexcept = default;

                /**
                 * Instatiates a new status object.
                 * @param builtin The MPI status built-in object.
                 */
                inline status(const raw_type& builtin) noexcept
                :   m_raw {builtin}
                {}

                inline status& operator=(const status&) noexcept = default;
                inline status& operator=(status&&) noexcept = default;

                /**
                 * Converts to the built-in status object.
                 * @return The built-in status object.
                 */
                inline operator const raw_type&() const noexcept
                {
                    return m_raw;
                }

                /**
                 * Retrieves the message error code.
                 * @return The error code.
                 */
                inline error_code error() const noexcept
                {
                    return m_raw.MPI_ERROR;
                }

                /**
                 * Retrieves the source of the message.
                 * @return The message source node.
                 */
                inline mpi::node source() const noexcept
                {
                    return m_raw.MPI_SOURCE;
                }

                /**
                 * Retrieves the message tag.
                 * @return The retrieved message tag.
                 */
                inline mpi::tag tag() const noexcept
                {
                    return m_raw.MPI_TAG;
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
                    MPI_Get_count(&m_raw, datatype::get<T>(), &value);
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
                    MPI_Test_cancelled(&m_raw, &flag);
                    return (flag != 0);
                }
        };

        /**
         * The last operation's status.
         * @see mpi::status
         */
        extern status last_status;

        /**
         * Represents an incoming or outcoming message payload of collective
         * communication operations. In practice, this object serves as the
         * base of a neutral context state for messages.
         * @tparam T The payload's element type.
         * @since 0.1.1
         */
        template <typename T>
        class payload : public buffer<T>
        {
            protected:
                using underlying_type = buffer<T>;                      /// The payload's underlying type.

            public:
                using element_type = typename buffer<T>::element_type;  /// The payload's element type.
                using pointer_type = typename buffer<T>::pointer_type;  /// The buffer's pointer type.
                using return_type = payload<T>; /// The payload's return type of collective operations.

            public:
                inline payload() noexcept = default;
                inline payload(const payload&) noexcept = default;
                inline payload(payload&&) noexcept = default;

                /**
                 * Creates a new payload from simple object value.
                 * @param value The payload's value.
                 */
                inline payload(element_type& value) noexcept
                :   underlying_type {pointer_type::weak(&value), 1}
                {}

                /**
                 * Creates a new payload from already existing buffer.
                 * @param ptr The payload's buffer pointer.
                 * @param size The payload's buffer size.
                 */
                inline payload(element_type *ptr, size_t size = 1) noexcept
                :   underlying_type {pointer_type::weak(ptr), size}
                {}

                /**
                 * Creates a new payload based on a pre-allocated buffer.
                 * @param buf The buffer to create payload from.
                 */
                inline payload(underlying_type&& buf) noexcept
                :   underlying_type {std::move(buf)}
                {}

                /**
                 * Handles any memory transfers, swaps or deallocations needed
                 * for an effective clean-up of this payload object.
                 */
                inline virtual ~payload() noexcept = default;

                inline payload& operator=(const payload&) = default;
                inline payload& operator=(payload&&) = default;

                /**
                 * Converts the payload into an instance of the element type,
                 * this is specially useful for operations with singletons.
                 * @return The converted payload to the element type.
                 */
                inline operator element_type() noexcept
                {
                    return underlying_type::operator[](0);
                }

                /**
                 * Converts the payload into an instance of the element type,
                 * this is specially useful for operations with singletons.
                 * @return The converted payload to the element type.
                 */
                inline operator const element_type() const noexcept
                {
                    return underlying_type::operator[](0);
                }

                /**
                 * Gives access to raw payload buffer's data. This is needed because
                 * unfortunatelly, all MPI functions take non-const pointers.
                 * @return The payload buffer's internal pointer.
                 */
                inline auto raw() const noexcept -> element_type *
                {
                    return const_cast<element_type *>(underlying_type::raw());
                }

                /**
                 * Unfortunately, all MPI functions take a simple integer as their
                 * arguments. Therefore, we must return an integer to satisfy them.
                 * @return The payload's buffer length.
                 */
                inline auto size() const noexcept -> int
                {
                    return static_cast<int>(underlying_type::size());
                }

                /**
                 * Gets the type identification of the payload's element type.
                 * @return The payload's element type id.
                 */
                inline auto type() const noexcept -> msa::mpi::datatype::id
                {
                    return msa::mpi::datatype::get<element_type>();
                }

                /**
                 * Copies data from an existing buffer instance.
                 * @param buf The target buffer to copy data from.
                 * @return A newly created payload instance.
                 */
                static inline payload copy(const underlying_type& buf)
                {
                    return payload {underlying_type::copy(buf)};
                }

                /**
                 * Copies data from a vector instance.
                 * @param vector The target vector instance to copy data from.
                 * @return A newly created payload instance.
                 */
                static inline payload copy(const std::vector<element_type>& vector)
                {
                    return payload {underlying_type::copy(vector)};
                }

                /**
                 * Copies data from an existing pointer.
                 * @param ptr The target pointer to copy from.
                 * @param count The number of elements to copy.
                 * @return A newly created payload instance.
                 */
                static inline payload copy(const element_type *ptr, size_t count)
                {
                    return payload {underlying_type::copy(ptr, count)};
                }

                /**
                 * Creates a new payload by copying a single value.
                 * @param value The value to be copied to new payload.
                 * @return A newly created payload instance.
                 */
                static inline payload copy(const element_type& value)
                {
                    return copy(&value, 1);
                }

                /**
                 * Creates a new payload of given size.
                 * @param size The payload's number of elements.
                 * @return The newly created payload instance.
                 */
                static inline payload make(size_t size = 1) noexcept
                {
                    return payload {underlying_type::make(size)};
                }
        };

        /**
         * A message payload buffer context for STL vectors.
         * @tparam T The message payload type.
         * @since 0.1.1
         */
        template <typename T>
        class payload<std::vector<T>> : public payload<T>
        {
            public:
                using return_type = payload<T>; /// The payload's return type of collective operations.

            public:
                /**
                 * Creates a new payload from STL vector.
                 * @param ref The payload as a STL vector.
                 */
                inline payload(std::vector<T>& ref) noexcept
                :   payload<T> {ref.data(), ref.size()}
                {}
        };

        /**
         * A message payload buffer context for... buffers.
         * @tparam T The message payload type.
         * @since 0.1.1
         */
        template <typename T>
        class payload<buffer<T>> : public payload<T>
        {
            public:
                using return_type = payload<T>; /// The payload's return type of collective operations.

            public:
                /**
                 * Creates a new payload from buffer.
                 * @param ref The buffer to use as payload.
                 */
                inline payload(buffer<T>& ref) noexcept
                :   payload<T> {ref.raw(), ref.size()}
                {}
        };

        namespace op
        {
            /**
             * Identifies an operator for MPI collective operations.
             * @since 0.1.1
             */
            using functor = MPI_Op;

            /**
             * Keeps track of all user defined operator's functors created during execution.
             * @since 0.1.1
             */
            extern std::vector<functor> ref_op;

            /**
             * Maps a datatype to an user-created operator. This is necessary because
             * it is almost technically impossible to inject the operator inside the
             * wrapper without an extremelly convoluted mechanism.
             * @since 0.1.1
             */
            extern std::map<functor, void *> op_list;

            /**
             * Informs the currently active operator. This will be useful for injecting
             * the correct operator inside the wrapper.
             * @since 0.1.1
             */
            extern functor active;

            /**
             * Wraps an operator transforming it into an MPI operator.
             * @tparam T The type the operator works onto.
             * @param a The operation's first operand.
             * @param b The operation's second operand and output value.
             * @param len The number of elements in given operation.
             */
            template <typename T>
            void fwrap(void *a, void *b, int *len, MPI_Datatype *)
            {
                using function_type = typename utils::op<T>::function_type;
                auto f = reinterpret_cast<function_type>(op_list[active]);

                for(int i = 0; i < *len; ++i)
                    static_cast<T*>(b)[i] = f(static_cast<T*>(a)[i], static_cast<T*>(b)[i]);
            }

            /**
             * Creates a new MPI operator from user function.
             * @tparam T The type the operator works onto.
             * @param fop The functor of operator to be created.
             * @param commutative Is the operator commutative?
             * @return The identifier for operator created.
             */
            template <typename T>
            inline functor create(const utils::op<T>& fop, bool commutative = true)
            {
                functor result;
                check(MPI_Op_create(fwrap<T>, commutative, &result));
                op_list[result] = reinterpret_cast<void *>(&fop);
                ref_op.push_back(result);
                return result;
            }

            /**#@+
             * Listing of built-in operators. The use of these instead of creating
             * new operators is highly recommended.
             * @since 0.1.1
             */
            static functor const& max      = MPI_MAX;
            static functor const& min      = MPI_MIN;
            static functor const& add      = MPI_SUM;
            static functor const& mul      = MPI_PROD;
            static functor const& andl     = MPI_LAND;
            static functor const& andb     = MPI_BAND;
            static functor const& orl      = MPI_LOR;
            static functor const& orb      = MPI_BOR;
            static functor const& xorl     = MPI_LXOR;
            static functor const& xorb     = MPI_BXOR;
            static functor const& minloc   = MPI_MINLOC;
            static functor const& maxloc   = MPI_MAXLOC;
            static functor const& replace  = MPI_REPLACE;
            /**#@-*/
        }

        /**
         * Represents an internal MPI communicator, which allows communication
         * and synchronization among a set of nodes and processes.
         * @since 0.1.1
         */
        class communicator
        {
            protected:
                using raw_type = MPI_Comm;      /// The internal MPI communicator type.

            private:
                uint32_t m_size = 0;            /// The number of nodes in communicator.
                mpi::node m_rank = 0;           /// The current node's rank in communicator.
                MPI_Comm m_raw = MPI_COMM_NULL; /// The internal MPI communicator reference.

            public:
                inline communicator() noexcept = default;
                inline communicator(const communicator&) noexcept = default;
                inline communicator(communicator&&) noexcept = default;

                inline communicator& operator=(const communicator&) noexcept = default;
                inline communicator& operator=(communicator&&) noexcept = default;

                /**
                 * Allows the communicator instance to be seen as an internal MPI
                 * communicator type seamlessly.
                 * @return The interal communicator pointer.
                 */
                inline operator raw_type() const noexcept
                {
                    return m_raw;
                }

                /**
                 * Informs the current node's rank in relation to current communicator.
                 * @return The node's rank in communicator.
                 */
                inline const mpi::node& rank() const noexcept
                {
                    return m_rank;
                }

                /**
                 * Informs the total number of nodes in current communicator.
                 * @return The number of nodes in communicator.
                 */
                inline const uint32_t& size() const noexcept
                {
                    return m_size;
                }

                /**
                 * Builds up a new communicator instance from built-in type.
                 * @param comm Built-in communicator instance.
                 * @return The new communicator instance.
                 */
                static inline auto build(MPI_Comm comm) -> communicator
                {
                    int rank, size;
                    check(MPI_Comm_rank(comm, &rank));
                    check(MPI_Comm_size(comm, &size));
                    return {static_cast<uint32_t>(size), rank, comm};
                }

                /**
                 * Splits nodes into different communicators according to selected color.
                 * @param comm The original communicator to be split.
                 * @param color The color selected by current node.
                 * @param key The key used to assigned a node id in new communicator.
                 * @return The obtained communicator from split operation.
                 */
                static inline auto split(const communicator& comm, int color, int key = any) -> communicator
                {
                    MPI_Comm newcomm;
                    check(MPI_Comm_split(comm, color, (key > 0 ? key : comm.m_rank), &newcomm));
                    return build(newcomm);
                }

                /**
                 * Cleans up the resources used by communicator.
                 * @param comm The communicator to be destroyed.
                 */
                static inline void free(communicator& comm)
                {
                    check(MPI_Comm_free(&comm.m_raw));
                    comm.m_raw = MPI_COMM_NULL;
                }

            private:
                /**
                 * Initializes a new communicator instance.
                 * @param size The number of nodes in given communicator.
                 * @param rank The current node's rank in communicator.
                 * @param raw The internal communicator reference.
                 */
                inline communicator(const uint32_t& size, const mpi::node& rank, raw_type raw)
                :   m_size {size}
                ,   m_rank {rank}
                ,   m_raw {raw}
                {}
        };

        /**
         * The default communicator instance.
         * @see mpi::communicator
         */
        extern communicator world;

        /**
         * Global MPI initialization and finalization routines. These functions
         * must be respectively called before and after any and every MPI operations.
         * @since 0.1.1
         */
        extern void init(int&, char **&);
        extern void finalize();

        /**
         * Synchronizes all nodes in a communicator.
         * @param comm The communicator the operation should apply to.
         */
        inline void barrier(const communicator& comm = world)
        {
            check(MPI_Barrier(comm));
        }

        /**#@+
         * Broadcasts data to all nodes in given communicator.
         * @tparam T Type of buffer data to broadcast.
         * @param data The target buffer to broadcast.
         * @param size The number of buffer's elements to broadcast.
         * @param root The operation's root node.
         * @param comm The communicator this operation applies to.
         * @return The broadcast message payload.
         */
        template <typename T>
        inline typename payload<T>::return_type broadcast(
                payload<T>& load
            ,   const node& root = msa::node::master
            ,   const communicator& comm = world
            )
        {
            check(MPI_Bcast(load.raw(), load.size(), load.type(), root, comm));
            return load;
        }

        template <typename T>
        inline typename payload<T>::return_type broadcast(
                T *data
            ,   size_t size = 1
            ,   const node& root = msa::node::master
            ,   const communicator& comm = world
            )
        {
            auto load = payload<T>::copy(data, size);
            return mpi::broadcast(load, root, comm);
        }

        template <typename T>
        inline typename payload<T>::return_type broadcast(
                T& data
            ,   const node& root = msa::node::master
            ,   const communicator& comm = world
            )
        {
            auto load = payload<T> {data};
            auto size = load.size();

            size = mpi::broadcast(&size, 1, root, comm);

            if(comm.rank() != root)
                load = payload<T>::make(size);

            return mpi::broadcast(load, root, comm);
        }
        /**#@-*/

        /**
         * Inspects incoming message and retrieves its status.
         * @param src The source node.
         * @param tag The identifying message tag.
         * @param comm The communicator this operation applies to.
         * @return The inspected message status.
         */
        inline status probe(const node& src = any, const mpi::tag& tag = any, const communicator& comm = world)
        {
            status::raw_type stt;
            check(MPI_Probe(src, tag, comm, &stt));
            return last_status = status {stt};
        }

        /**#@+
         * Sends data to a node connected to the cluster.
         * @tparam T Type of buffer data to send.
         * @param data The buffer to send.
         * @param size The number of buffer's elements to send.
         * @param dest The destination node.
         * @param tag The identifying message tag.
         * @param comm The communicator this operation applies to.
         */
        template <typename T>
        inline void send(
                const payload<T>& load
            ,   const node& dest = msa::node::master
            ,   const mpi::tag& tag = any
            ,   const communicator& comm = world
            )
        {
            check(MPI_Send(load.raw(), load.size(), load.type(), dest, tag < 0 ? MPI_TAG_UB : tag, comm));
        }

        template <typename T>
        inline void send(
                T *data
            ,   size_t size = 1
            ,   const node& dest = msa::node::master
            ,   const mpi::tag& tag = any
            ,   const communicator& comm = world
            )
        {
            mpi::send(payload<T> {data, size}, dest, tag, comm);
        }

        template <typename T>
        inline void send(
                T& data
            ,   const node& dest = msa::node::master
            ,   const mpi::tag& tag = any
            ,   const communicator& comm = world
            )
        {
            mpi::send(payload<T> {data}, dest, tag, comm);
        }
        /**#@-*/

        /**#@+
         * Receives data from a node connected to the cluster.
         * @tparam T Type of buffer data to receive.
         * @param src The source node.
         * @param tag The identifying tag.
         * @param comm The communicator this operation applies to.
         * @return The received message payload.
         */
        template <typename T>
        inline typename payload<T>::return_type receive(
                payload<T>& load
            ,   const node& src = any
            ,   const mpi::tag& tag = any
            ,   const communicator& comm = world
            )
        {
            status::raw_type stt;
            check(MPI_Recv(load.raw(), load.size(), load.type(), src, tag, comm, &stt));
            last_status = status {stt};
            return load;
        }

        template <typename T>
        inline typename payload<T>::return_type receive(
                const node& src = any
            ,   const mpi::tag& tag = any
            ,   const communicator& comm = world
            )
        {
            auto info = mpi::probe(src, tag, comm);
            using E = typename payload<T>::element_type;
            auto load = payload<T>::make(info.count<E>());
            return mpi::receive(load, src, tag, comm);
        }
        /**#@-*/

        /**#@+
         * Gathers value from all processes to a single value and send to all processes.
         * @tparam T The type of buffer data to reduce.
         * @param data The node's outgoing buffer.
         * @param size The outgoing and incoming buffers' sizes.
         * @param fop The operation's reducing function.
         * @param comm The communicator this operation applies to.
         * @return The reduced value payload.
         */
        template <typename T>
        inline typename payload<T>::return_type allreduce(
                const payload<T>& out
            ,   const op::functor& fop
            ,   const communicator& comm = world
            )
        {
            auto in = payload<T>::make(out.size());
            check(MPI_Allreduce(out.raw(), in.raw(), out.size(), out.type(), op::active = fop, comm));
            return in;
        }

        template <typename T>
        inline typename payload<T>::return_type allreduce(
                T *data
            ,   size_t size
            ,   const op::functor& fop
            ,   const communicator& comm = world
            )
        {
            return mpi::allreduce(payload<T> {data, size}, fop, comm);
        }

        template <typename T>
        inline typename payload<T>::return_type allreduce(
                T& data
            ,   const op::functor& fop
            ,   const communicator& comm = world
            )
        {
            return mpi::allreduce(payload<T> {data}, fop, comm);
        }
        /**#@-*/

        /**#@+
         * Gathers value from all processes to a single value and send to root process.
         * @tparam T The type of buffer data to reduce.
         * @param data The node's outgoing buffer.
         * @param size The outgoing and incoming buffers' sizes.
         * @param fop The operation's reducing function.
         * @param root The operation's root process.
         * @param comm The communicator this operation applies to.
         * @return The reduced value payload.
         */
        template <typename T>
        inline typename payload<T>::return_type reduce(
                const payload<T>& out
            ,   const op::functor& fop
            ,   const node& root = msa::node::master
            ,   const communicator& comm = world
            )
        {
            auto in = payload<T>::make(out.size());
            check(MPI_Reduce(out.raw(), in.raw(), out.size(), out.type(), op::active = fop, root, comm));
            return in;
        }

        template <typename T>
        inline typename payload<T>::return_type reduce(
                T *data
            ,   size_t size
            ,   const op::functor& fop
            ,   const node& root = msa::node::master
            ,   const communicator& comm = world
            )
        {
            return mpi::reduce(payload<T> {data, size}, fop, root, comm);
        }

        template <typename T>
        inline typename payload<T>::return_type reduce(
                T& data
            ,   const op::functor& fop
            ,   const node& root = msa::node::master
            ,   const communicator& comm = world
            )
        {
            return mpi::reduce(payload<T> {data}, fop, root, comm);
        }
        /**#@-*/

        /**#@+
         * Gathers data from all nodes and deliver combined data to all nodes
         * @tparam T The type of buffer data to gather.
         * @param data The outgoing buffer.
         * @param size The outgoing buffer's size.
         * @param comm The communicator this operation applies to.
         * @return The gathered message payload.
         */
        template <typename T>
        inline typename payload<T>::return_type allgather(
                const payload<T>& out
            ,   const communicator& comm = world
            )
        {
            auto in = payload<T>::make(out.size() * comm.size());
            check(MPI_Allgather(out.raw(), out.size(), out.type(), in.raw(), out.size(), out.type(), comm));
            return in;
        }

        template <typename T>
        inline typename payload<T>::return_type allgather(
                const payload<T>& out
            ,   const payload<int>& size
            ,   const payload<int>& displ
            ,   const communicator& comm = world
            )
        {
            auto type = out.type();
            auto in = payload<T>::make(std::accumulate(size.begin(), size.end(), 0));
            check(MPI_Allgatherv(out.raw(), out.size(), type, in.raw(), size.raw(), displ.raw(), type, comm));
            return in;
        }

        template <typename T>
        inline typename payload<T>::return_type allgather(
                T *data
            ,   size_t size = 1
            ,   const communicator& comm = world
            )
        {
            auto load = payload<T> {data, size};
            auto sizeall = mpi::allgather(payload<int>::copy(size), comm);
            
            if(std::all_of(sizeall.begin(), sizeall.end(), [size](int i) { return i == size; }))
                return mpi::allgather(load, comm);

            auto dispall = std::vector<int> (comm.size() + 1);
            std::partial_sum(sizeall.begin(), sizeall.end(), dispall.begin() + 1);
            return mpi::allgather(load, sizeall, payload<decltype(dispall)> {dispall}, comm);
        }

        template <typename T>
        inline typename payload<T>::return_type allgather(
                T& data
            ,   const communicator& comm = world
            )
        {
            auto load = payload<T> {data};
            return mpi::allgather(load.raw(), load.size(), comm);
        }
        /**#@-*/

        /**#@+
         * Gathers together values from a group of processes
         * @tparam T The type of buffer data to gather.
         * @param data The outgoing buffer.
         * @param size The outgoing buffer's size.
         * @param root The operation's target node.
         * @param comm The communicator this operation applies to.
         * @return The gathered message payload.
         */
        template <typename T>
        inline typename payload<T>::return_type gather(
                const payload<T>& out
            ,   const node& root = msa::node::master
            ,   const communicator& comm = world
            )
        {
            auto in = payload<T>::make(out.size() * comm.size());
            check(MPI_Gather(out.raw(), out.size(), out.type(), in.raw(), out.size(), out.type(), root, comm));
            return in;
        }

        template <typename T>
        inline typename payload<T>::return_type gather(
                const payload<T>& out
            ,   const payload<int>& size
            ,   const payload<int>& displ
            ,   const node& root = msa::node::master
            ,   const communicator& comm = world
            )
        {
            auto type = out.type();
            auto in = payload<T>::make(std::accumulate(size.begin(), size.end(), 0));
            check(MPI_Gatherv(out.raw(), out.size(), type, in.raw(), size.raw(), displ.raw(), type, root, comm));
            return in;
        }

        template <typename T>
        inline typename payload<T>::return_type gather(
                T *data
            ,   size_t size = 1
            ,   const node& root = msa::node::master
            ,   const communicator& comm = world
            )
        {
            auto load = payload<T> {data, size};
            auto sizeall = mpi::allgather(payload<int>::copy(size), comm);

            if(std::all_of(sizeall.begin(), sizeall.end(), [size](int i) { return i == size; }))
                return mpi::gather(load, root, comm);

            auto dispall = std::vector<int> (comm.size() + 1);
            std::partial_sum(sizeall.begin(), sizeall.end(), dispall.begin() + 1);
            return mpi::gather(load, sizeall, payload<decltype(dispall)> {dispall}, root, comm);
        }

        template <typename T>
        inline typename payload<T>::return_type gather(
                T& data
            ,   const node& root = msa::node::master
            ,   const communicator& comm = world
            )
        {
            auto load = payload<T> {data};
            return mpi::gather(load.raw(), load.size(), root, comm);
        }
        /**#@-*/

        /**#@+
         * Sends data from one process to all other processes in a communicator
         * @tparam T The type of buffer data to scatter.
         * @param data The outgoing buffer.
         * @param size The outgoing buffer's size.
         * @param root The operation's root node.
         * @param comm The communicator this operation applies to.
         * @return The scathered message payload.
         */
        template <typename T>
        inline typename payload<T>::return_type scatter(
                const payload<T>& out
            ,   const node& root = msa::node::master
            ,   const communicator& comm = world
            )
        {
            auto in = payload<T>::make(out.size());
            check(MPI_Scatter(out.raw(), in.size(), out.type(), in.raw(), in.size(), out.type(), root, comm));
            return in;
        }

        template <typename T>
        inline typename payload<T>::return_type scatter(
                const payload<T>& out
            ,   const payload<int>& size
            ,   const payload<int>& displ
            ,   const node& root = msa::node::master
            ,   const communicator& comm = world
            )
        {
            auto type = out.type();
            auto in = payload<T>::make(size[comm.rank()]);
            check(MPI_Scatterv(out.raw(), size.raw(), displ.raw(), type, in.raw(), in.size(), type, root, comm));
            return in;
        }

        template <typename T>
        inline typename payload<T>::return_type scatter(
                T *data
            ,   size_t size
            ,   const node& root = msa::node::master
            ,   const communicator& comm = world
            )
        {
            size = mpi::broadcast(size, root, comm);

            size_t quotient  = size / comm.size();
            int remainder = size % comm.size();

            if(remainder == 0)
                return mpi::scatter(payload<T> {data, quotient}, root, comm);

            auto sizeall = payload<int>::make(comm.size());
            auto dispall = payload<int>::make(comm.size());

            for(int i = 0, n = comm.size(); i < n; ++i) {
                sizeall[i] = quotient + (remainder > i);
                dispall[i] = quotient * i + utils::min(i, remainder);
            }

            return mpi::scatter(payload<T> {data}, sizeall, dispall, root, comm);
        }

        template <typename T>
        inline typename payload<T>::return_type scatter(
                T& data
            ,   const node& root = msa::node::master
            ,   const communicator& comm = world
            )
        {
            auto load = payload<T> {data};
            return mpi::scatter(load.raw(), load.size(), root, comm);
        }
        /**#@-*/
    }

    /**
     * Checks whether two communicators are the same.
     * @param a The first communicator to compare.
     * @param b The second communicator to compare.
     * @return Are both communicators the same?
     */
    inline bool operator==(const mpi::communicator& a, const mpi::communicator& b) noexcept
    {
        return ((MPI_Comm)(a)) == ((MPI_Comm)(b));
    }

    /**
     * Checks whether two communicators are different.
     * @param a The first communicator to compare.
     * @param b The second communicator to compare.
     * @return Are both communicators different?
     */
    inline bool operator!=(const mpi::communicator& a, const mpi::communicator& b) noexcept
    {
        return ((MPI_Comm)(a)) != ((MPI_Comm)(b));
    }
}

#endif