/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements a wrapper around MPI functions and structures.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <map>
#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>

#include "node.hpp"
#include "buffer.hpp"
#include "exception.hpp"
#include "environment.h"
#include "reflection.hpp"

#if !defined(__museqa_runtime_cython)

#include <mpi.h>

namespace museqa
{
    namespace mpi
    {
        /**
         * Represents the identifier of a node connected to the cluster.
         * @since 0.1.1
         */
        using node = museqa::node::id;

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
        class exception : public museqa::exception
        {
            protected:
                using underlying_type = museqa::exception; /// The underlying exception.

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
            using E = museqa::mpi::exception;
            enforce<E>(code == error::success, code);
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
            template <> inline id get<int16_t>()  { return MPI_INT16_T; };
            template <> inline id get<int32_t>()  { return MPI_INT32_T; };
            template <> inline id get<int64_t>()  { return MPI_INT64_T; };
            template <> inline id get<uint8_t>()  { return MPI_UINT8_T; };
            template <> inline id get<uint16_t>() { return MPI_UINT16_T; };
            template <> inline id get<uint32_t>() { return MPI_UINT32_T; };
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
                inline static void build(int *blocks, MPI_Aint *offsets, MPI_Datatype *types)
                {
                    using reflection = museqa::reflection_tuple<T>;
                    type_builder<reflection>::template build<T>(blocks, offsets, types);
                }
            };

            /**
             * The middle recursion steps for creating a new datatype.
             * @tparam T The current type in reflection tuple to become part of the datatype.
             * @tparam U The following types in reflection tuple.
             * @since 0.1.1
             */
            template <typename T, typename ...U>
            struct type_builder<museqa::tuple<T, U...>>
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
                inline static void build(int *blocks, MPI_Aint *offsets, MPI_Datatype *types)
                {
                    type_builder<museqa::tuple<T>>::template build<O, N>(blocks, offsets, types);
                    type_builder<museqa::tuple<U...>>::template build<O, N + 1>(blocks, offsets, types);
                }
            };

            /**
             * The final recursion step for creating a new datatype.
             * @param T The last type in reflection tuple to become part of the datatype.
             * @since 0.1.1
             */
            template <typename T>
            struct type_builder<museqa::tuple<T>>
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
                inline static void build(int *blocks, MPI_Aint *offsets, MPI_Datatype *types)
                {
                    offsets[N] = museqa::reflection<O>::template offset<N>();
                    types[N]   = museqa::mpi::datatype::get<museqa::base<T>>();
                    blocks[N]  = museqa::utils::max((int) std::extent<T>::value, 1);
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
            extern std::vector<datatype::id> ref_type;

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
                    datatype::id m_typeid;          /// The raw MPI datatype reference.

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

                        detail::mpi::type_builder<T>::build(blocks, offsets, types);

                        mpi::check(MPI_Type_create_struct(count, blocks, offsets, types, &m_typeid));
                        mpi::check(MPI_Type_commit(&m_typeid));
                        ref_type.push_back(m_typeid);
                    }

                    /**
                     * Gives access to the datatype created for the requested type.
                     * @return The datatype created for type T.
                     */
                    inline static datatype::id get()
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
                 * Determines the number of elements contained on the message.
                 * @tparam T The message content type.
                 * @return The number of elements contained in the message.
                 */
                template <typename T>
                inline int32_t count() const noexcept
                {
                    return count(datatype::get<T>());
                }

                /**
                 * Determines the number of elements contained on the message.
                 * @param type_id The id of the content type to count.
                 * @return The number of elements contained in the message.
                 */
                inline int32_t count(const datatype::id& type_id) const noexcept
                {
                    int value;
                    MPI_Get_count(&m_raw, type_id, &value);
                    return value != MPI_UNDEFINED ? value : -1;
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
         * Represents a type-agnostic message, which can be more easily dealt with
         * when using MPI functions.
         * @since 0.1.1
         */
        struct message
        {
            using pointer_type = pointer<void>;

            pointer_type ptr;       /// The message's contents pointer.
            int32_t size = 1;       /// The total number of elements in message.

            datatype::id type;      /// The id of the message's type.
            size_t typesize;        /// The message's type size in bytes.

            inline message() noexcept = default;
            inline message(const message&) noexcept = default;
            inline message(message&&) noexcept = default;

            /**
             * Initializes a new message from a generic pointer type.
             * @tparam T The message's elements' type.
             * @param ptr The pointer for the message payload.
             * @param size The total number of elements in message.
             */
            template <typename T>
            inline message(pointer<T>& ptr, size_t size = 1) noexcept
            :   ptr {ptr}
            ,   size {static_cast<int32_t>(size)}
            ,   type {datatype::get<pure<T>>()}
            ,   typesize {sizeof(pure<T>)}
            {}

            /**
             * Initializes a new message base on another message and a new pointer.
             * @param msg The message to base the new message on.
             * @param ptr The pointer for the message payload.
             * @param size The total number of elements on the pointer.
             */
            inline message(const message& msg, pointer_type& ptr, size_t size = 1)
            :   ptr {ptr}
            ,   size {static_cast<int32_t>(size)}
            ,   type {msg.type}
            ,   typesize {msg.typesize}
            {}

            inline message& operator=(const message&) noexcept = default;
            inline message& operator=(message&&) noexcept = default;

            /**
             * Creates a new message with the given size based on another one.
             * @param msg The message to base the new message on.
             * @param size The total number of elements on the new message.
             */
            static inline message make(const message& msg, size_t size = 1) noexcept
            {
                pointer_type ptr = pointer<char[]>::make(size * msg.typesize);
                return message {msg, ptr, size};
            }
        };

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
                using return_type = payload<T>;                         /// The payload's operation return type.

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
                 * Creates a new payload from a collective operation message.
                 * @param msg The message to create the payload from.
                 */
                inline payload(message&& msg) noexcept
                :   underlying_type {std::move(msg.ptr), static_cast<size_t>(msg.size)}
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
                 * Converts the payload into a message for collective operations.
                 * @return The payload converted to a message.
                 */
                inline operator message() noexcept
                {
                    return message {this->m_ptr, this->m_size};
                }

                /**
                 * Converts the payload into a vector instance with the payload's
                 * elements. Thus, one can use vectors through MPI seamlessly.
                 * @return The converted payload to a vector.
                 */
                inline operator std::vector<element_type>() noexcept
                {
                    return std::vector<element_type> (this->begin(), this->end());
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
                /**
                 * Creates a new payload from STL vector.
                 * @param ref The payload as a STL vector.
                 */
                inline payload(std::vector<T>& ref) noexcept
                :   payload<T> {ref.data(), ref.size()}
                {}

                using payload<T>::payload;
                using payload<T>::operator=;
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
                /**
                 * Creates a new payload from buffer.
                 * @param ref The buffer to use as payload.
                 */
                inline payload(buffer<T>& ref) noexcept
                :   payload<T> {ref.raw(), ref.size()}
                {}

                using payload<T>::payload;
                using payload<T>::operator=;
        };

        namespace op
        {
            /**
             * Identifies an operator for MPI collective operations.
             * @since 0.1.1
             */
            using id = MPI_Op;

            /**
             * Keeps track of all user defined operator's functors created during execution.
             * @since 0.1.1
             */
            extern std::vector<op::id> ref_op;

            /**
             * Maps a datatype to an user-created operator. This is necessary because
             * it is almost technically impossible to inject the operator inside the
             * wrapper without an extremelly convoluted mechanism.
             * @since 0.1.1
             */
            extern std::map<op::id, void *> op_list;

            /**
             * Informs the currently active operator. This will be useful for injecting
             * the correct operator inside the wrapper.
             * @since 0.1.1
             */
            extern op::id active;

            /**
             * Wraps an operator transforming it into an MPI operator.
             * @tparam T The type the operator works onto.
             * @param a The operation's first operand.
             * @param b The operation's second operand and output value.
             * @param len The number of elements in given operation.
             */
            template <typename T>
            void fwrap(void *a, void *b, int *len, datatype::id *)
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
            inline op::id create(const utils::op<T>& fop, bool commutative = true)
            {
                op::id result;
                mpi::check(MPI_Op_create(fwrap<T>, commutative, &result));
                op_list[result] = reinterpret_cast<void *>(&fop);
                ref_op.push_back(result);
                return result;
            }

            /**#@+
             * Listing of built-in operators. The use of these instead of creating
             * new operators is highly recommended.
             * @since 0.1.1
             */
            static op::id const& max      = MPI_MAX;
            static op::id const& min      = MPI_MIN;
            static op::id const& add      = MPI_SUM;
            static op::id const& mul      = MPI_PROD;
            static op::id const& andl     = MPI_LAND;
            static op::id const& andb     = MPI_BAND;
            static op::id const& orl      = MPI_LOR;
            static op::id const& orb      = MPI_BOR;
            static op::id const& xorl     = MPI_LXOR;
            static op::id const& xorb     = MPI_BXOR;
            static op::id const& minloc   = MPI_MINLOC;
            static op::id const& maxloc   = MPI_MAXLOC;
            static op::id const& replace  = MPI_REPLACE;
            /**#@-*/
        }

        /**
         * Represents an internal MPI communicator, which allows communication
         * and synchronization among a set of nodes and processes.
         * @since 0.1.1
         */
        class communicator
        {
            public:
                using raw_type = MPI_Comm;      /// The internal MPI communicator type.

            private:
                mpi::node m_rank = 0;           /// The current node's rank in communicator.
                uint32_t m_size = 0;            /// The number of nodes in communicator.
                raw_type m_raw = MPI_COMM_NULL; /// The internal MPI communicator reference.

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

                static auto build(raw_type) -> communicator;
                static auto split(const communicator&, int, int = any) -> communicator;
                static void free(communicator&);

            private:
                /**
                 * Initializes a new communicator instance.
                 * @param rank The current node's rank in communicator.
                 * @param size The number of nodes in given communicator.
                 * @param raw The internal communicator reference.
                 */
                inline communicator(const mpi::node& rank, const uint32_t& size, raw_type raw)
                :   m_rank {rank}
                ,   m_size {size}
                ,   m_raw {raw}
                {}
        };

        /**
         * The default communicator instance.
         * @see mpi::communicator
         */
        extern communicator world;

        /*
         * Global MPI initialization and finalization routines. These functions
         * must be respectively called before and after any and every MPI operations.
         */
        extern void init(int&, char **&);
        extern void finalize();

        /**
         * Synchronizes all nodes in a communicator.
         * @param comm The communicator the operation should apply to.
         */
        inline void barrier(const communicator& comm = world)
        {
            mpi::check(MPI_Barrier(comm));
        }

        /**
         * Broadcasts a message to all nodes in given communicator.
         * @param out The message to broadcast.
         * @param root The operation's root node.
         * @param comm The communicator this operation applies to.
         * @return The message that has been received.
         */
        inline message broadcast(message out, node root, const communicator& comm)
        {
            message in = comm.rank() != root ? message::make(out, out.size) : out;
            mpi::check(MPI_Bcast(in.ptr, in.size, in.type, root, comm));
            return in;
        }

        /**
         * Sends a message to a specific node connected to the cluster.
         * @param out The message to send.
         * @param dest The destination node.
         * @param tag The identifying message tag.
         * @param comm The communicator this operation applies to.
         */
        inline void send(message out, node dest, mpi::tag tag, const communicator& comm)
        {
            mpi::tag msgtag = tag >= 0 ? tag : MPI_TAG_UB;
            mpi::check(MPI_Send(out.ptr, out.size, out.type, dest, msgtag, comm));
        }

        /**
         * Receives a message from a specific node connected to the cluster.
         * @param in The message to receive.
         * @param src The source node.
         * @param tag The identifying tag.
         * @param comm The communicator this operation applies to.
         * @return The received message payload.
         */
        inline message receive(message in, node src, mpi::tag tag, const communicator& comm)
        {
            status::raw_type stat;
            mpi::check(MPI_Recv(in.ptr, in.size, in.type, src, tag, comm, &stat));
            last_status = status {stat};
            return in;
        }

        /**
         * Reduces a message from all nodes to a single one on all nodes.
         * @param out The node's outgoing message.
         * @param fop The operation's reducing function.
         * @param comm The communicator this operation applies to.
         * @return The reduced message.
         */
        inline message allreduce(message out, const op::id& fop, const communicator& comm)
        {
            message in = message::make(out, out.size);
            mpi::check(MPI_Allreduce(out.ptr, in.ptr, in.size, in.type, op::active = fop, comm));
            return in;
        }

        /**
         * Reduces a message from all nodes to a single one on root node.
         * @param out The node's outgoing message.
         * @param fop The operation's reducing function.
         * @param root The operation's root process.
         * @param comm The communicator this operation applies to.
         * @return The reduced message.
         */
        inline message reduce(message out, const op::id& fop, node root, const communicator& comm)
        {
            message in = message::make(out, out.size);
            mpi::check(MPI_Reduce(out.ptr, in.ptr, in.size, in.type, op::active = fop, root, comm));
            return in;
        }

        /**
         * Gathers messages from all nodes and delivers as one to all nodes.
         * @param out The node's outgoing message.
         * @param comm The communicator this operation applies to.
         * @return The gathered message.
         */
        inline message allgather(message out, const communicator& comm)
        {
            message in = message::make(out, out.size * comm.size());
            mpi::check(MPI_Allgather(out.ptr, out.size, in.type, in.ptr, out.size, in.type, comm));
            return in;
        }

        /**
         * Gathers different messages from all nodes and delivers as one to all nodes.
         * @param out The node's outgoing message.
         * @param msize The number of elements to be sent by each node.
         * @param mdisp The displacement of each node's messages.
         * @param comm The communicator this operation applies to.
         * @return The gathered message.
         */
        inline message allgatherv(message out, message msize, message mdisp, const communicator& comm)
        {
            const int *size = static_cast<const int *>(&msize.ptr);
            const int *disp = static_cast<const int *>(&mdisp.ptr);
            message in = message::make(out, std::accumulate(size, size + comm.size(), 0));
            mpi::check(MPI_Allgatherv(out.ptr, out.size, in.type, in.ptr, size, disp, in.type, comm));
            return in;
        }

        /**
         * Gathers messages from all nodes and delivers as one to the root node.
         * @param out The node's outgoing message.
         * @param root The operation's target node.
         * @param comm The communicator this operation applies to.
         * @return The gathered message.
         */
        inline message gather(message out, node root, const communicator& comm)
        {
            message in = message::make(out, comm.rank() != root ? 0 : out.size * comm.size());
            mpi::check(MPI_Gather(out.ptr, out.size, in.type, in.ptr, out.size, in.type, root, comm));
            return in;
        }

        /**
         * Gathers different messages from all nodes and delivers as one to the root node.
         * @param out The node's outgoing message.
         * @param msize The number of elements to be sent by each node.
         * @param mdisp The displacement of each node's messages.
         * @param root The operation's target node.
         * @param comm The communicator this operation applies to.
         * @return The gathered message.
         */
        inline message gatherv(message out, message msize, message mdisp, node root, const communicator& comm)
        {
            const int *size = static_cast<const int *>(&msize.ptr);
            const int *disp = static_cast<const int *>(&mdisp.ptr);
            message in = message::make(out, comm.rank() != root ? 0 : std::accumulate(size, size + comm.size(), 0));
            mpi::check(MPI_Gatherv(out.ptr, out.size, in.type, in.ptr, size, disp, in.type, root, comm));
            return in;
        }

        /**
         * Scatters a message from one node to all other nodes in a communicator.
         * @param out The outgoing message.
         * @param root The operation's root node.
         * @param comm The communicator this operation applies to.
         * @return The scattered message.
         */
        inline message scatter(message out, node root, const communicator& comm)
        {
            message in = message::make(out, out.size / comm.size());
            mpi::check(MPI_Scatter(out.ptr, in.size, in.type, in.ptr, in.size, in.type, root, comm));
            return in;
        }

        /**
         * Scatters a message from one node to all other nodes in a communicator.
         * @param out The outgoing message.
         * @param msize The number of elements to be sent to each node.
         * @param mdisp The displacement of each node's messages.
         * @param root The operation's root node.
         * @param comm The communicator this operation applies to.
         * @return The scattered message.
         */
        inline message scatterv(message out, message msize, message mdisp, node root, const communicator& comm)
        {
            const int *size = static_cast<const int *>(&msize.ptr);
            const int *disp = static_cast<const int *>(&mdisp.ptr);
            message in = message::make(out, size[comm.rank()]);
            mpi::check(MPI_Scatterv(out.ptr, size, disp, in.type, in.ptr, in.size, in.type, root, comm));
            return in;
        }

        /**#@+
         * Broadcasts generic data to all nodes in a given communicator.
         * @tparam T Type of buffer data to broadcast.
         * @param data The target buffer to broadcast.
         * @param size The number of buffer's elements to broadcast.
         * @param root The operation's root node.
         * @param comm The communicator this operation applies to.
         * @return The message payload that has been broadcast.
         */
        template <typename T>
        inline typename payload<T>::return_type broadcast(
                T *data
            ,   size_t size = 1
            ,   node root = museqa::node::master
            ,   const communicator& comm = world
            )
        {
            message msg = payload<T> {data, size};
            return mpi::broadcast(msg, root, comm);
        }

        template <typename T>
        inline typename payload<T>::return_type broadcast(
                T& data
            ,   node root = museqa::node::master
            ,   const communicator& comm = world
            )
        {
            message msg = payload<T> {data};
            msg.size = mpi::broadcast(&msg.size, 1, root, comm);
            return mpi::broadcast(msg, root, comm);
        }
        /**#@-*/

        /**
         * Inspects incoming message and retrieves its status.
         * @param src The source node.
         * @param tag The identifying message tag.
         * @param comm The communicator this operation applies to.
         * @return The inspected message status.
         */
        inline status probe(node src = any, mpi::tag tag = any, const communicator& comm = world)
        {
            status::raw_type stt;
            mpi::check(MPI_Probe(src, tag, comm, &stt));
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
                T *data
            ,   size_t size = 1
            ,   node dest = museqa::node::master
            ,   mpi::tag tag = any
            ,   const communicator& comm = world
            )
        {
            message msg = payload<T> {data, size};
            mpi::send(msg, dest, tag, comm);
        }

        template <typename T>
        inline void send(
                T& data
            ,   node dest = museqa::node::master
            ,   mpi::tag tag = any
            ,   const communicator& comm = world
            )
        {
            message msg = payload<T> {data};
            mpi::send(msg, dest, tag, comm);
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
                node src = any
            ,   mpi::tag tag = any
            ,   const communicator& comm = world
            )
        {
            auto stat = mpi::probe(src, tag, comm);
            using E = typename payload<T>::element_type;
            message msg = payload<T>::make(stat.count<E>());
            return mpi::receive(msg, src, tag, comm);
        }
        /**#@-*/

        /**#@+
         * Reduces value from all nodes to a single value and send to all nodes.
         * @tparam T The type of buffer data to reduce.
         * @param data The node's outgoing buffer.
         * @param size The outgoing and incoming buffers' sizes.
         * @param fop The operation's reducing function.
         * @param comm The communicator this operation applies to.
         * @return The reduced value payload.
         */
        template <typename T>
        inline typename payload<T>::return_type allreduce(
                T *data
            ,   size_t size
            ,   const op::id& fop
            ,   const communicator& comm = world
            )
        {
            message msg = payload<T> {data, size};
            return mpi::allreduce(msg, fop, comm);
        }

        template <typename T>
        inline typename payload<T>::return_type allreduce(
                T& data
            ,   const op::id& fop
            ,   const communicator& comm = world
            )
        {
            message msg = payload<T> {data};
            return mpi::allreduce(msg, fop, comm);
        }
        /**#@-*/

        /**#@+
         * Reduces value from all nodes to a single value and send to root node.
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
                T *data
            ,   size_t size
            ,   const op::id& fop
            ,   node root = museqa::node::master
            ,   const communicator& comm = world
            )
        {
            message msg = payload<T> {data, size};
            return mpi::reduce(msg, fop, root, comm);
        }

        template <typename T>
        inline typename payload<T>::return_type reduce(
                T& data
            ,   const op::id& fop
            ,   node root = museqa::node::master
            ,   const communicator& comm = world
            )
        {
            message msg = payload<T> {data};
            return mpi::reduce(msg, fop, root, comm);
        }
        /**#@-*/

        /**#@+
         * Gathers data from all nodes and deliver combined data to all nodes.
         * @tparam T The type of buffer data to gather.
         * @param data The outgoing buffer.
         * @param size The outgoing buffer's size.
         * @param comm The communicator this operation applies to.
         * @return The gathered message payload.
         */
        template <typename T>
        inline typename payload<T>::return_type allgather(
                T *data
            ,   size_t size = 1
            ,   const communicator& comm = world
            )
        {
            message msg = payload<T> {data, size};
            return mpi::allgather(msg, comm);
        }

        template <typename T>
        inline typename payload<T>::return_type allgather(
                T& data
            ,   const communicator& comm = world
            )
        {
            message msg = payload<T> {data};

            payload<int> size = mpi::allgather(&msg.size, 1, comm);
            payload<int> disp = payload<int>::make(comm.size() + 1);

            bool equal = true;

            for(size_t i = 0, n = comm.size(); i < n; ++i) {
                equal = equal && (size[0] == size[i]);
                disp[i + 1] = disp[i] + size[i];
            }

            if(equal) return mpi::allgather(msg, comm);
            else return mpi::allgatherv(msg, size, disp, comm);
        }
        /**#@-*/

        /**#@+
         * Gathers data from all nodes and deliver it all combined to root node.
         * @tparam T The type of buffer data to gather.
         * @param data The outgoing buffer.
         * @param size The outgoing buffer's size.
         * @param root The operation's target node.
         * @param comm The communicator this operation applies to.
         * @return The gathered message payload.
         */
        template <typename T>
        inline typename payload<T>::return_type gather(
                T *data
            ,   size_t size = 1
            ,   node root = museqa::node::master
            ,   const communicator& comm = world
            )
        {
            message msg = payload<T> {data, size};
            return mpi::gather(msg, root, comm);
        }

        template <typename T>
        inline typename payload<T>::return_type gather(
                T& data
            ,   node root = museqa::node::master
            ,   const communicator& comm = world
            )
        {
            message msg = payload<T> {data};

            payload<int> size = mpi::allgather(&msg.size, 1, comm);
            payload<int> disp = payload<int>::make(comm.size() + 1);

            bool equal = true;

            for(size_t i = 0, n = comm.size(); i < n; ++i) {
                equal = equal && (size[0] == size[i]);
                disp[i + 1] = disp[i] + size[i];
            }

            if(equal) return mpi::gather(msg, root, comm);
            else return mpi::gatherv(msg, size, disp, root, comm);
        }
        /**#@-*/

        /**#@+
         * Sends data from one node to all other nodes in a communicator
         * @tparam T The type of buffer data to scatter.
         * @param data The outgoing buffer.
         * @param size The outgoing buffer's size.
         * @param root The operation's root node.
         * @param comm The communicator this operation applies to.
         * @return The scattered message payload.
         */
        template <typename T>
        inline typename payload<T>::return_type scatter(
                T *data
            ,   size_t size = 1
            ,   node root = museqa::node::master
            ,   const communicator& comm = world
            )
        {
            message msg = payload<T> {data, size};
            return mpi::scatter(msg, root, comm);
        }

        template <typename T>
        inline typename payload<T>::return_type scatter(
                T& data
            ,   node root = museqa::node::master
            ,   const communicator& comm = world
            )
        {
            message msg = payload<T> {data};
            msg.size = mpi::broadcast(&msg.size, 1, root, comm);

            int quotient  = msg.size / comm.size();
            int remainder = msg.size % comm.size();

            if(!remainder) return mpi::scatter(msg, root, comm);

            payload<int> size = payload<int>::make(comm.size());
            payload<int> disp = payload<int>::make(comm.size());

            for(int i = 0, n = comm.size(); i < n; ++i) {
                size[i] = quotient + (remainder > i);
                disp[i] = quotient * i + utils::min(i, remainder);
            }

            return mpi::scatterv(msg, size, disp, root, comm);
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
        return ((mpi::communicator::raw_type)(a)) == ((mpi::communicator::raw_type)(b));
    }

    /**
     * Checks whether two communicators are different.
     * @param a The first communicator to compare.
     * @param b The second communicator to compare.
     * @return Are both communicators different?
     */
    inline bool operator!=(const mpi::communicator& a, const mpi::communicator& b) noexcept
    {
        return ((mpi::communicator::raw_type)(a)) != ((mpi::communicator::raw_type)(b));
    }
}

#endif
