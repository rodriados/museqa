/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A general type-agnostic message payload for MPI collective operations.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#if !defined(MUSEQA_AVOID_MPI)

#include <vector>
#include <cstdint>
#include <utility>

#include <museqa/utility.hpp>
#include <museqa/mpi/message.hpp>
#include <museqa/memory/buffer.hpp>
#include <museqa/memory/pointer/weak.hpp>

namespace museqa
{
    namespace mpi
    {
        /**
         * Represents an incoming or outcoming message payload transiting through
         * a collective operation between nodes via MPI. In practice, this object
         * serves as a neutral context for messages of different types.
         * @tparam T The payload's message contents type.
         * @since 1.0
         */
        template <typename T>
        class payload : public memory::buffer<T>
        {
            static_assert(std::is_trivial<T>::value, "MPI messages must have trivial types");

          protected:
            typedef memory::buffer<T> underlying_type;

          public:
            typedef payload<T> return_type;         /// The return type of collective operations.
            typedef typename underlying_type::element_type element_type;
            
          public:
            inline payload() noexcept = default;
            inline payload(const payload&) noexcept = default;
            inline payload(payload&&) noexcept = default;

            /**
             * Wraps a simple value into a payload instance.
             * @param value The payload's message simple contents.
             */
            inline payload(element_type& value) noexcept
              : payload {&value, 1}
            {}

            /**
             * Wraps a raw pointer into a message payload.
             * @param ptr The raw message buffer pointer.
             * @param size The total number of elements in message.
             */
            inline payload(element_type *ptr, size_t size = 1) noexcept
              : underlying_type {memory::pointer::weak<element_type[]>{ptr}, size}
            {}

            /**
             * Wraps a buffer into a message payload.
             * @param buffer The buffer to wrap into the payload.
             */
            inline payload(underlying_type&& buffer) noexcept
              : underlying_type {std::forward<decltype(buffer)>(buffer)}
            {}

            /**
             * Wraps an MPI message into a payload.
             * @param msg The message to create payload from.
             */
            inline payload(message&& msg) noexcept
              : underlying_type {std::move(msg.ptr), static_cast<size_t>(msg.size)}
            {}

            inline virtual ~payload() noexcept = default;

            inline payload& operator=(const payload&) = default;
            inline payload& operator=(payload&&) = default;

            /**
             * Converts the payload into a type-less message instance.
             * @return The converted message instance.
             */
            inline operator message() noexcept
            {
                return message {this->m_ptr, this->m_capacity};
            }

            /**
             * Seamlessly converts the payload into its message contents type.
             * @return The payload's message contents.
             */
            inline operator element_type&() noexcept(museqa::unsafe)
            {
                return underlying_type::operator[](0);
            }

            /**
             * Converts the payload into a vector instance by copying the payload's
             * contents. Thus, one can seamlessly use vectors through MPI.
             * @return The converted payload to a vector.
             */
            inline operator std::vector<element_type>() noexcept
            {
                return std::vector<element_type> (this->begin(), this->end());
            }
        };

        /**
         * Represents a message payload context for buffers.
         * @tparam T The payload's message type.
         * @since 1.0
         */
        template <typename T>
        class payload<memory::buffer<T>> : public payload<T>
        {
          public:
            /**
             * Wraps an existing buffer into a message payload.
             * @param buffer The buffer to be wrapped.
             */
            inline payload(memory::buffer<T>& buffer) noexcept
              : payload<T> {buffer.begin(), buffer.capacity()}
            {}

            using payload<T>::operator=;
        };

        /**
         * Represents a message payload context for vectors.
         * @tparam T The payload's message type.
         * @since 1.0
         */
        template <typename T>
        class payload<std::vector<T>> : public payload<T>
        {
          public:
            /**
             * Wraps a vector into a message payload.
             * @param vector The vector to be wrapped.
             */
            inline payload(std::vector<T>& vector) noexcept
              : payload<T> {vector.data(), vector.size()}
            {}

            using payload<T>::operator=;
        };
    }

    namespace factory
    {
        namespace mpi
        {
            /**
             * Allocates a new payload with the requested elements capacity.
             * @tparam T The message's contents type.
             * @param capacity The expected number of elements in payload.
             * @return The new allocated message payload.
             */
            template <typename T>
            inline auto payload(size_t capacity = 1) noexcept -> museqa::mpi::payload<T>
            {
                return factory::buffer<T>(capacity);
            }

            /**
             * Copies data from a raw pointer into a MPI message payload.
             * @tparam T The message's contents type.
             * @param ptr The target pointer to copy data from.
             * @param count The number of elements to be copied.
             * @return The new message payload with copied elements.
             */
            template <typename T>
            inline auto payload(const T* ptr, size_t count = 1) noexcept -> museqa::mpi::payload<T>
            {
                return factory::buffer(ptr, count);
            }

            /**
             * Copies one value into a MPI message payload.
             * @tparam T The message's contents type.
             * @param value The value to be copied into a new payload.
             * @return The new message payload with the copied element.
             */
            template <typename T>
            inline auto payload(const T& value) noexcept -> museqa::mpi::payload<T>
            {
                return factory::mpi::payload(&value, 1);
            }

            /**
             * Copies data from a buffer into a MPI message payload.
             * @tparam T The message's contents type.
             * @param buffer The buffer instance to copy contents from.
             * @return The new message payload with copied elements.
             */
            template <typename T>
            inline auto payload(const memory::buffer<T>& buffer) noexcept -> museqa::mpi::payload<T>
            {
                return factory::buffer(buffer);
            }

            /**
             * Copies data from a vector into a MPI message payload.
             * @tparam T The message's contents type.
             * @param vector The vector instance to copy contents from.
             * @return The new message payload with copied elements.
             */
            template <typename T>
            inline auto payload(const std::vector<T>& vector) noexcept -> museqa::mpi::payload<T>
            {
                return factory::buffer(vector);
            }
        }
    }
}

#endif
