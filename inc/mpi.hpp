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

                /**
                 * Instatiates a new status object.
                 * @param builtin The MPI status built-in object.
                 */
                inline status(const raw_type& builtin) noexcept
                :   m_raw {builtin}
                {}

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
    }

    namespace mpi
    {
        namespace payload
        {
            /**
             * Represents an incoming or outcoming message payload of collective
             * communication operations. In practice, this object serves as the
             * base of a neutral context state for messages.
             * @since 0.1.1
             */
            class base
            {
                protected:
                    void *m_ptr = nullptr;      /// The payload's raw pointer.
                    size_t m_size = 0;          /// The payload's buffer size.

                public:
                    inline base() noexcept = delete;
                    inline base(const base&) noexcept = default;
                    inline base(base&&) noexcept = default;

                    /**
                     * Creates a new payload from already existing buffer.
                     * @param ptr The payload's buffer pointer.
                     * @param size The payload's buffer size.
                     */
                    inline base(void *ptr, size_t size = 1) noexcept
                    :   m_ptr {ptr}
                    ,   m_size {size}
                    {}

                    /**
                     * Handles any memory transfers, swaps or deallocations needed
                     * for an effective clean-up of this payload object.
                     */
                    inline virtual ~base() noexcept = default;

                    inline base& operator=(const base&) noexcept = default;
                    inline base& operator=(base&&) noexcept = default;

                    /**
                     * Retrieves the payload's buffer pointer.
                     * @return The payload's buffer pointer.
                     */
                    inline void *raw() const noexcept
                    {
                        return m_ptr;
                    }

                    /**
                     * Retrieves the payload's buffer capacity.
                     * @return The payload's size or capacity.
                     */
                    inline size_t size() const noexcept
                    {
                        return m_size;
                    }

                    virtual auto resize(size_t) -> void * = 0;
                    virtual auto type() -> datatype::id = 0;
            };

            /**
             * A message payload buffer context for scalar types.
             * @tparam T The message payload buffer type.
             * @since 0.1.1
             */
            template <typename T>
            class buffer : public base
            {
                public:
                    using element_type = pure<T>;       /// The payload's elementary type.

                public:
                    /**
                     * Creates a new payload from simple object value.
                     * @param value The payload's value.
                     */
                    inline buffer(element_type& value) noexcept
                    :   base {static_cast<void *>(&value), 1}
                    {}

                    /**
                     * Creates a new payload from already existing buffer.
                     * @param ptr The payload's buffer pointer.
                     * @param size The payload's buffer size.
                     */
                    inline buffer(element_type *ptr, size_t size = 1) noexcept
                    :   base {static_cast<void *>(ptr), size}
                    {}

                    /**
                     * Retrieves the payload's buffer pointer.
                     * @return The payload's buffer pointer.
                     */
                    inline element_type *raw() const noexcept
                    {
                        return static_cast<element_type *>(this->m_ptr);
                    }

                    /**
                     * Creates a new pointer with given size and swaps buffers.
                     * This allows the payload to receive an incoming message.
                     * @param (ignored) The new minimum payload size.
                     * @return The resized buffer pointer.
                     * @throw mpi::exception Cannot resize simple pointers.
                     */
                    inline auto resize(size_t) -> void * override
                    {
                        throw mpi::exception {"cannot resize simple pointers"};
                    }

                    /**
                     * Gets the type identification of the payload's element type.
                     * @return The payload's element type id.
                     */
                    inline auto type() -> datatype::id override
                    {
                        return datatype::get<element_type>();
                    }
            };

            /**
             * A message payload buffer context for STL vectors.
             * @tparam T The message payload type.
             * @since 0.1.1
             */
            template <typename T>
            class buffer<std::vector<T>> : public buffer<T>
            {
                protected:
                    std::vector<T>& m_ref;          /// The original vector's reference.

                public:
                    /**
                     * Creates a new payload from STL vector.
                     * @param ref The payload as a STL vector.
                     */
                    inline buffer(std::vector<T>& ref) noexcept
                    :   buffer<T> {ref.data(), ref.size()}
                    ,   m_ref {ref}
                    {}

                    /**
                     * Creates a new vector and swaps buffers.
                     * @param size The new minimum payload capacity.
                     * @return The resized buffer pointer.
                     */
                    inline auto resize(size_t size) -> void * override
                    {
                        if(this->m_size < size)
                            m_ref.resize(this->m_size = size);
                        return static_cast<void *>(this->m_ptr = m_ref.data());
                    }
            };

            /**
             * A message payload buffer context for MSA buffers.
             * @tparam T The message payload type.
             * @since 0.1.1
             */
            template <typename T>
            class buffer<msa::buffer<T>> : public buffer<T>
            {
                protected:
                    msa::buffer<T>& m_ref;          /// The original buffer's reference.

                public:
                    /**
                     * Creates a new payload from buffer.
                     * @param ref The buffer to use as payload.
                     */
                    inline buffer(msa::buffer<T>& ref) noexcept
                    :   buffer<T> {ref.raw(), ref.size()}
                    ,   m_ref {ref}
                    {}

                    /**
                     * Creates a new buffer object and swaps contents.
                     * @param size The new minimum payload capacity.
                     * @return The new buffer pointer
                     */
                    inline auto resize(size_t size) -> void * override
                    {
                        if(this->m_size != size)
                            m_ref = msa::buffer<T>::make(m_ref.allocator(), this->m_size = size);
                        return static_cast<void *>(this->m_ptr = m_ref.raw());
                    }
            };

            /**#@+
             * Creates a new payload from given buffer type.
             * @tparam T The given buffer content type.
             * @param tgt The base payload buffer.
             * @param ptr The base payload pointer.
             * @param size The base payload element count.
             * @return The new payload.
             */
            template <typename T>
            inline auto make(T& tgt) -> payload::buffer<T>
            {
                return payload::buffer<T> {tgt};
            }

            template <typename T>
            inline auto make(T *ptr, size_t size) -> payload::buffer<T>
            {
                return payload::buffer<T> {ptr, size};
            }
            /**#@-*/
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
                using function_type = typename utils::op<T>::functor;
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
            static constexpr id const& add      = MPI_SUM;
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

        /**
         * Represents an internal MPI communicator, which allows communication
         * and synchronization among a set of nodes and processes.
         * @since 0.1.1
         */
        class communicator
        {
            protected:
                using raw_type = MPI_Comm;      /// The communicator internal MPI type.

            protected:
                raw_type m_ref = MPI_COMM_NULL; /// The internal MPI communicator reference.

            public:
                const node rank = 0;            /// The rank of current node in relation to communicator.
                const int32_t size = 0;         /// The number of nodes known by communicator.

            public:
                inline communicator() noexcept = default;
                inline communicator(const communicator&) noexcept = default;
                inline communicator(communicator&&) noexcept = default;

                inline communicator& operator=(const communicator&) noexcept = delete;
                inline communicator& operator=(communicator&&) noexcept = delete;

                void barrier() const;
                void broadcast(payload::base *, const node&) const;

                void send(payload::base *, const node&, const tag&) const;
                auto receive(payload::base *, const node&, const tag&) const -> status;
                auto probe(const node&, const tag&) const -> status;

                void gather(payload::base *, payload::base *, const node&) const;
                void gatherv(payload::base *, int *, payload::base *, const node&) const;
                void allgather(payload::base *, payload::base *) const;
                void scatter(payload::base *, payload::base *, const node&) const;

                /**
                 * Builds up a new communicator instance from built-in type.
                 * @param ref The internal communicator reference.
                 * @return The new communicator instance.
                 */
                inline static auto build(const raw_type& ref) -> communicator
                {
                    return communicator {ref};
                }

                /**
                 * Splits nodes into different communicators according to selected color.
                 * @param comm The original communicator to be split.
                 * @param color The color selected by current node.
                 * @param key The key used to assigned a node id in new communicator.
                 * @return The obtained communicator from split operation.
                 */
                inline static auto split(const communicator& comm, int32_t color, int32_t key = any)
                -> communicator
                {
                    raw_type new_ref;
                    check(MPI_Comm_split(comm.m_ref, color, (key > 0 ? key : comm.rank), &new_ref));
                    return communicator {new_ref};
                }

                /**
                 * Cleans up resources used by communicator.
                 * @param comm The communicator to be destroyed.
                 */
                inline static void free(const communicator& comm)
                {
                    check(MPI_Comm_free(&comm.m_ref));
                    comm.m_ref = MPI_COMM_NULL;
                }

            private:
                communicator(const raw_type&);
        };

        /**
         * The default communicator instance.
         * @see mpi::communicator::id
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
            comm.barrier();
        }

        /**#@+
         * Broadcasts data to all nodes in given communicator.
         * @tparam T Type of buffer data to broadcast.
         * @param data The target buffer to broadcast.
         * @param size The number of buffer's elements to broadcast.
         * @param root The operation's root node.
         * @param comm The communicator this operation applies to.
         */
        template <typename T>
        inline void broadcast(
                T *data
            ,   int size = 1
            ,   const node& root = msa::node::master
            ,   const communicator& comm = world
            )
        {
            auto load = payload::make(data, size);
            comm.broadcast(&load, root);
        }

        template <typename T>
        inline void broadcast(
                T& data
            ,   const node& root = msa::node::master
            ,   const communicator& comm = world
            )
        {
            auto load = payload::make(data);
            auto size = load.size();

            broadcast(&size, 1, root, comm);
            broadcast(load.resize(size), size, root, comm);
        }
        /**#@-*/

        /**#@+
         * Sends data to a node connected to the cluster.
         * @tparam T Type of buffer data to send.
         * @param data The buffer to send.
         * @param size The number of buffer's elements to send.
         * @param dest The destination node.
         * @param tag The identifying message tag.
         * @param comm The communicator this operation applies to.
         * @return MPI error code if not successful.
         */
        template <typename T>
        inline void send(
                T *data
            ,   int size = 1
            ,   const node& dest = msa::node::master
            ,   const mpi::tag& tag = MPI_TAG_UB
            ,   const communicator& comm = world
            )
        {
            auto load = payload::make(data, size);
            comm.send(&load, dest, tag);
        }

        template <typename T>
        inline void send(
                T& data
            ,   const node& dest = msa::node::master
            ,   const mpi::tag& tag = MPI_TAG_UB
            ,   const communicator& comm = world
            )
        {
            auto load = payload::make(data);
            comm.send(&load, dest, tag);
        }
        /**#@-*/

        /**#@+
         * Receives data from a node connected to the cluster.
         * @tparam T Type of buffer data to receive.
         * @param data The buffer to receive data into.
         * @param size The number of buffer's elements to receive.
         * @param src The source node.
         * @param tag The identifying tag.
         * @param comm The communicator this operation applies to.
         * @return The message status
         */

        template <typename T>
        inline status receive(
                T *data
            ,   int size = 1
            ,   const node& src = any
            ,   const mpi::tag& tag = MPI_TAG_UB
            ,   const communicator::id& comm = world
            )
        {
            auto load = payload::make(data, size);
            return comm.receive(&load, src, tag);
        }

        template <typename T>
        inline status receive(
                T& data
            ,   const node& src = any
            ,   const mpi::tag& tag = MPI_TAG_UB
            ,   const communicator::id& comm = world
            )
        {
            auto load = payload::make(load);
            using P = typename decltype(load)::element_type;

            auto size = comm.probe(src, tag).count<P>();
            return receive(load.resize(size), size, src, tag, comm);
        }
        /**#@-*/

        /**
         * Inspects incoming message and retrieves its status.
         * @param src The source node.
         * @param tag The identifying message tag.
         * @param comm The communicator this operation applies to.
         * @return The inspected message status.
         */
        status probe(const node& src = any, const mpi::tag& tag = any, const communicator& comm = world)
        {
            return comm.probe(src, tag);
        }

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
        template <typename T, typename U>
        inline void gather(
                T *odata, int osize
            ,   U *idata, int isize
            ,   const node& root = msa::node::master
            ,   const communicator& comm = world
            )
        {
            auto oload = payload::make(odata, osize);
            auto iload = payload::make(idata, isize);
            comm.gather(&oload, &iload, root);
        }

        template <typename T, typename U>
        inline void gather(
                T *odata, int osize,
            ,   U *idata, int isize, int *idispl
            ,   const node& root = msa::node::master
            ,   const communicator& comm = world
            )
        {
            auto oload = payload::make(odata, osize);
            auto iload = payload::make(idata, isize);
            comm.gatherv(&oload, &iload, idispl, root);
        }

        template <typename T, typename U>
        inline void gather(
                T& odata
            ,   U& idata
            ,   const node& root = msa::node::master
            ,   const communicator& comm = world
            )
        {
            auto oload = payload::make(odata);
            auto iload = payload::make(idata);
            comm.gather(&oload, &iload, root);
        }

        template <typename T, typename U>
        inline void gather(
                T& odata
            ,   U& idata, int *idispl
            ,   const node& root = msa::node::master
            ,   const communicator& comm = world
            )
        {
            auto oload = payload::make(odata);
            auto iload = payload::make(idata);
            comm.gatherv(&oload, &iload, idispl, root);
        }




        template <typename T, typename U>
        inline void gather(
                T& odata, int osize, int odispl
            ,   U& idata
            ,   const node& root = msa::node::master
            ,   const communicator& comm = world
            )
        {
            int32_t isize;
            auto oload = payload::make(odata);
            auto iload = payload::make(idata);
            
            reduce(osize, isize, mpi::op::add, root, comm);
            gather(&odispl, 1, ldispl.data(), 1, root, comm);
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
         * Gathers data from all nodes and deliver combined data to all nodes
         * @tparam T The type of buffer data to gather.
         * @tparam U The type of buffer data to gather.
         * @param out_data The outgoing buffer.
         * @param out_count The outgoing buffer size.
         * @param in_data The incoming buffer.
         * @param in_count The size of incoming buffer from each node.
         * @param displ The data displacement of each node.
         * @param comm The communicator this operation applies to.
         */


        template <typename T>
        inline void allgather(
                T *out_data, int out_count
            ,   T *in_data, int in_count
            ,   const communicator::id& comm = world
            )
        {
            check(MPI_Allgather(
                    out_data, out_count, datatype::get<T>()
                ,   in_data, in_count, datatype::get<T>(), comm.ref
                ));
        }

        template <typename T>
        inline void allgather(
                T *out_data, int out_count, int *displ
            ,   T *in_data, int *in_count
            ,   const communicator::id& comm = world
            )
        {
            check(MPI_Allgatherv(
                    out_data, out_count, datatype::get<T>()
                ,   in_data, in_count, displ, datatype::get<T>(), comm.ref
                ));
        }

        template <typename T, typename U>
        inline void allgather(
                T& out_data, int out_count, int displ
            ,   U& in_data
            ,   const communicator::id& comm = world
            )
        {
            auto to_send = mpi::payload(out_data);
            auto to_recv = mpi::payload(in_data);
            using S = typename decltype(to_send)::element_type;
            using R = typename decltype(to_recv)::element_type;
            static_assert(std::is_same<S, R>::value, "cannot gather with different types");

            std::vector<int> all_count(comm.size), all_displ(comm.size);
            allgather(&out_count, 1, all_count.data(), 1, comm);
            allgather(&displ, 1, all_displ.data(), 1, comm);

            to_recv.resize(std::accumulate(all_count.begin(), all_count.end(), 0));
            allgather(to_send.data(), to_send.size(), to_recv.data(), all_count.data(), all_displ.data(), comm);
        }

        template <typename T, typename U>
        inline void allgather(T& out_data, U& in_data, const communicator::id& comm = world)
        {
            auto to_send = mpi::payload(out_data);
            auto to_recv = mpi::payload(in_data);
            using S = typename decltype(to_send)::element_type;
            using R = typename decltype(to_recv)::element_type;
            static_assert(std::is_same<S, R>::value, "cannot gather with different types");

            int count = to_send.size();
            std::vector<int> all_count(comm.size), all_displ(comm.size + 1);

            allgather(&count, 1, all_count.data(), 1, comm);

            bool equal = std::all_of(all_count.begin(), all_count.end(), [&count](int i) { return i == count; });
            std::partial_sum(all_count.begin(), all_count.end(), all_displ.begin() + 1);

            to_recv.resize(all_displ.back());

            if(equal) allgather(to_send.data(), count, to_recv.data(), count, comm);
            else allgather(to_send.data(), count, to_recv.data(), all_count.data(), all_displ.data(), comm);
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
}

#endif