/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A MPI message status wrapper implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#if !defined(MUSEQA_AVOID_MPI)

#include <mpi.h>

#include <cstdint>

#include <museqa/mpi/common.hpp>
#include <museqa/mpi/type.hpp>

namespace museqa
{
    namespace mpi
    {
        /**
         * Wraps a MPI message status from a received or receivable message.
         * @since 1.0
         */
        class status
        {
          public:
            typedef MPI_Status raw_type;                    /// The raw MPI status type.

          protected:
            mutable raw_type m_raw;                         /// The raw message status instance.

          public:
            inline status() noexcept = default;
            inline status(const status&) noexcept = default;
            inline status(status&&) noexcept = default;

            /**
             * Builds a new status from a raw status instance.
             * @param raw The raw MPI status object.
             */
            inline status(const raw_type& raw) noexcept
              : m_raw {raw}
            {}

            inline status& operator=(const status&) noexcept = default;
            inline status& operator=(status&&) noexcept = default;

            /**
             * Exposes the raw status object instance.
             * @return The raw MPI message status instance.
             */
            inline operator const raw_type&() const noexcept
            {
                return m_raw;
            }

            /**
             * Retrieves the status internal error code.
             * @return The MPI message status code.
             */
            inline auto code() const noexcept -> error::code
            {
                return m_raw.MPI_ERROR;
            }

            /**
             * Retrieves the status message's source node.
             * @return The message's source node.
             */
            inline auto source() const noexcept -> mpi::node
            {
                return m_raw.MPI_SOURCE;
            }

            /**
             * Retrieves the status message's tag.
             * @return The retrieved message's tag.
             */
            inline auto tag() const noexcept -> mpi::tag
            {
                return m_raw.MPI_TAG;
            }

            /**
             * Retrieves the number of elements contained in the message.
             * @tparam T The retrieved message's content type.
             * @return The number of elements contained in the message.
             */
            template <typename T>
            inline auto count() const noexcept -> int32_t
            {
                return count(type::identify<T>());
            }

            /**
             * Retrieves the number of elements contained in the message.
             * @param tid The id of the message's content type.
             * @return The number of elements contained in the message.
             */
            inline auto count(const type::id& tid) const noexcept -> int32_t
            {
                int value;
                MPI_Get_count(&m_raw, tid, &value);
                return MPI_UNDEFINED != value ? value : -1;
            }

            /**
             * Determines whether the message retrieval associated with the current
             * status instance has been cancelled.
             * @return Has the message been successfully cancelled?
             */
            inline auto cancelled() const noexcept -> bool
            {
                int flag;
                MPI_Test_cancelled(&m_raw, &flag);
                return 0 != flag;
            }
        };
    }
}

#endif
