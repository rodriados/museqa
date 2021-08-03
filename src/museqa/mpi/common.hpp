/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Miscellaneous utilities and functions for using MPI.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#if !defined(MUSEQA_AVOID_MPI)

#include <mpi.h>

#include <string>
#include <cstdint>
#include <utility>

#include <museqa/node.hpp>
#include <museqa/utility.hpp>
#include <museqa/exception.hpp>

namespace museqa
{
    namespace mpi
    {
        /**
         * The type for identifying a specific MPI-node within a cluster.
         * @since 1.0
         */
        using node = museqa::node::id;

        /**
         * The type for a message transmission tag.
         * @since 1.0
         */
        using tag = int32_t;

        /**
         * The value acceptable as a node source or a tag for a received message.
         * @since 1.0
         */
        enum : typename std::common_type<node, tag>::type { any = -1 };

        namespace error
        {
            /**
             * Indicates an MPI error code detected during execution.
             * @since 1.0
             */
            using code = decltype(MPI_SUCCESS);

            /**
             * Aliases for MPI error types and error codes.
             * @since 1.0
             */
            enum : error::code
            {
                success                 = MPI_SUCCESS
              , access                  = MPI_ERR_ACCESS
              , amode                   = MPI_ERR_AMODE
              , arg                     = MPI_ERR_ARG
              , assert                  = MPI_ERR_ASSERT
              , bad_file                = MPI_ERR_BAD_FILE
              , base                    = MPI_ERR_BASE
              , buffer                  = MPI_ERR_BUFFER
              , comm                    = MPI_ERR_COMM
              , conversion              = MPI_ERR_CONVERSION
              , count                   = MPI_ERR_COUNT
              , dims                    = MPI_ERR_DIMS
              , disp                    = MPI_ERR_DISP
              , dup_datarep             = MPI_ERR_DUP_DATAREP
              , file                    = MPI_ERR_FILE
              , file_exists             = MPI_ERR_FILE_EXISTS
              , file_in_use             = MPI_ERR_FILE_IN_USE
              , group                   = MPI_ERR_GROUP
              , in_status               = MPI_ERR_IN_STATUS
              , info                    = MPI_ERR_INFO
              , info_key                = MPI_ERR_INFO_KEY
              , info_nokey              = MPI_ERR_INFO_NOKEY
              , info_value              = MPI_ERR_INFO_VALUE
              , intern                  = MPI_ERR_INTERN
              , io                      = MPI_ERR_IO
              , keyval                  = MPI_ERR_KEYVAL
              , lastcode                = MPI_ERR_LASTCODE
              , locktype                = MPI_ERR_LOCKTYPE
              , name                    = MPI_ERR_NAME
              , no_mem                  = MPI_ERR_NO_MEM
              , no_space                = MPI_ERR_NO_SPACE
              , no_such_file            = MPI_ERR_NO_SUCH_FILE
              , not_same                = MPI_ERR_NOT_SAME
              , op                      = MPI_ERR_OP
              , other                   = MPI_ERR_OTHER
              , pending                 = MPI_ERR_PENDING
              , port                    = MPI_ERR_PORT
              , quota                   = MPI_ERR_QUOTA
              , rank                    = MPI_ERR_RANK
              , read_only               = MPI_ERR_READ_ONLY
              , request                 = MPI_ERR_REQUEST
              , rma_conflict            = MPI_ERR_RMA_CONFLICT
              , rma_sync                = MPI_ERR_RMA_SYNC
              , root                    = MPI_ERR_ROOT
              , service                 = MPI_ERR_SERVICE
              , size                    = MPI_ERR_SIZE
              , spawn                   = MPI_ERR_SPAWN
              , tag                     = MPI_ERR_TAG
              , topology                = MPI_ERR_TOPOLOGY
              , truncate                = MPI_ERR_TRUNCATE
              , type                    = MPI_ERR_TYPE
              , unsupported_datarep     = MPI_ERR_UNSUPPORTED_DATAREP
              , unsupported_operation   = MPI_ERR_UNSUPPORTED_OPERATION
              , win                     = MPI_ERR_WIN
              , unknown                 = MPI_ERR_UNKNOWN
            };

            /**
             * Produces an error message explaining a detected error.
             * @param err The error code to be explained.
             * @return The error description.
             */
            inline std::string describe(error::code err) noexcept
            {
                int _;
                char buffer[MPI_MAX_ERROR_STRING];

                return success != MPI_Error_string(err, buffer, &_)
                    ? "error while probing MPI eror message"
                    : buffer;
            }
        }

        /**
         * Represents a MPI-error detected during execution that can be thrown and
         * propagated through the code carrying an error message.
         * @since 1.0
         */
        class exception : public museqa::exception
        {
          protected:
            typedef museqa::exception underlying_type;      /// The underlying exception type.

          protected:
            error::code m_err;                              /// The error code detected during execution.

          public:
            /**
             * Builds a new exception instance.
             * @param err The error code reported by MPI.
             */
            inline exception(error::code err) noexcept
              : underlying_type {"mpi exception: {}", error::describe(err)}
              , m_err {err}
            {}

            /**
             * Builds a new exception instance from an error code.
             * @tparam T The format parameters' types.
             * @param err The error code.
             * @param fmtstr The exception message's format.
             * @param args The format's parameters.
             */
            template <typename ...T>
            inline exception(error::code err, const std::string& fmtstr, T&&... args) noexcept
              : underlying_type {fmtstr, std::forward<decltype(args)>(args)...}
              , m_err {err}
            {}

            using underlying_type::exception;

            /**
             * Retrieves the MPI error code wrapped by the exception.
             * @return The error code.
             */
            inline error::code code() const noexcept
            {
                return m_err;
            }
        };

        /**
         * Globally initiliazes the internal MPI machinery. This function must be
         * called before any other MPI calls or operations.
         * @since 1.0
         */
        extern void init(int&, char**&);

        /**
         * Finalizes and frees the resources used by MPI. No MPI function can be
         * called after this function is called.
         * @since 1.0.
         */
        extern void finalize();

        /**
         * Asserts whether the given condition is met and throws exception otherwise.
         * @tparam E The exception type to be raised in case of error.
         * @tparam T The exception's parameters' types.
         * @param condition The condition that must be evaluated as true.
         * @param params The assertion exception's parameters.
         */
        template <typename E = mpi::exception, typename ...T>
        inline void ensure(bool condition, T&&... params) noexcept(!safe)
        {
            museqa::ensure<E>(condition, std::forward<decltype(params)>(params)...);
        }

        /**
         * Checks whether a MPI function call or operation has been successful and
         * throws an exception otherwise.
         * @param err The error code obtained from the operation.
         * @throw The MPI error raised to an exception.
         */
        inline void check(error::code err) noexcept(!safe)
        {
            mpi::ensure(error::success == err, err);
        }
    }
}

#endif
