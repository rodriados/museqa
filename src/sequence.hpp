/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements an encoded biologic sequence storage.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <cstdint>

#include "utils.hpp"
#include "buffer.hpp"
#include "format.hpp"
#include "encoder.hpp"

namespace museqa
{
    /**
     * Holds an enconded sequence. The encoding pattern will used throughout all
     * steps: it saves up to a third of the required space and is easily revertable.
     * @since 0.1.1
     */
    class sequence : public encoder::buffer
    {
        protected:
            using underlying_buffer = encoder::buffer;      /// The underlying sequence buffer.

        public:
            static constexpr encoder::unit padding = encoder::end;

        public:
            inline sequence() noexcept = default;
            inline sequence(const sequence&) noexcept = default;
            inline sequence(sequence&&) noexcept = default;

            using underlying_buffer::buffer;

            /**
             * Initializes a new sequence from an instance of its underlying buffer.
             * @param buf The buffer to create the new sequence from.
             */
            inline sequence(const underlying_buffer& buf) noexcept
            :   underlying_buffer {buf}
            {}

            /**
             * Initializes a new compressed sequence.
             * @param ptr The pointer to buffer to be encoded.
             * @param size The buffer's size.
             */
            inline sequence(const char *ptr, size_t size) noexcept
            :   underlying_buffer {encoder::encode(ptr, size)}
            {}

            /**
             * Instantiates a new sequence.
             * @param str The string containing this sequence's data.
             */
            inline sequence(const std::string& str) noexcept
            :   sequence {str.data(), str.size()}
            {}

            /**
             * Instantiates a new sequence from a const string.
             * @tparam N The sequence's size.
             * @param str The string to initialize the new sequence.
             */
            template <size_t N>
            inline sequence(const char (&str)[N]) noexcept
            :   sequence {str, N - 1}
            {}

            inline sequence& operator=(const sequence&) = default;
            inline sequence& operator=(sequence&&) = default;

            /**
             * Retrieves the encoded unit at given offset.
             * @param offset The requested offset.
             * @return The unit in the specified offset.
             */
            __host__ __device__ inline encoder::unit operator[](ptrdiff_t offset) const
            {
                return encoder::access(*this, offset);
            }

            /**
             * Retrieves an encoded character block from sequence.
             * @param offset The index of the requested block.
             * @return The requested encoded block.
             */
            __host__ __device__ inline encoder::block block(ptrdiff_t offset) const
            {
                return underlying_buffer::operator[](offset);
            }

            /**
             * Informs the length of the sequence.
             * @return The sequence's length.
             */
            __host__ __device__ inline size_t length() const noexcept
            {
                return size() * encoder::block_size;
            }

            /**
             * Retrieves the sequence's unpadded length.
             * @return The sequence's length without any padding.
             */
            __host__ __device__ inline size_t unpadded() const noexcept
            {
                encoder::block last_block = block(size() - 1);
                size_t length = this->length();

                for(size_t i = 1; i < encoder::block_size; ++i)
                    length -= (padding == encoder::access(last_block, i));

                return length;
            }

            /**
             * Transforms the sequence into a string.
             * @return The sequence representation as a string.
             */
            __host__ __device__ inline std::string decode() const
            {
                return encoder::decode(*this);
            }
    };

    /**
     * Manages a slice of a sequence. The sequence must have already been initialized
     * and will have boundaries checked according to view pointers.
     * @since 0.1.1
     */
    class sequence_view : public buffer_slice<encoder::block>
    {
        protected:
            using underlying_buffer = buffer_slice<encoder::block>; /// The underlying sequence buffer.

        public:
            inline sequence_view() noexcept = default;
            inline sequence_view(const sequence_view&) noexcept = default;
            inline sequence_view(sequence_view&&) noexcept = default;

            using underlying_buffer::buffer_slice;

            inline sequence_view& operator=(const sequence_view&) = default; 
            inline sequence_view& operator=(sequence_view&&) = default;

            /**
             * Retrieves the encoded unit at given offset.
             * @param offset The requested offset.
             * @return The unit in the specified offset.
             */
            __host__ __device__ inline encoder::unit operator[](ptrdiff_t offset) const
            {
                return encoder::access(*this, offset);
            }

            /**
             * Retrieves an encoded character block from sequence.
             * @param offset The index of the requested block.
             * @return The requested encoded block.
             */
            __host__ __device__ inline encoder::block block(ptrdiff_t offset) const
            {
                return underlying_buffer::operator[](offset);
            }

            /**
             * Informs the length of the sequence.
             * @return The sequence's length.
             */
            __host__ __device__ inline size_t length() const noexcept
            {
                return size() * encoder::block_size;
            }

            /**
             * Transforms the sequence into a string.
             * @return The sequence representation as a string.
             */
            __host__ __device__ inline std::string decode() const
            {
                return encoder::decode(*this);
            }
    };

    namespace fmt
    {
        /**
         * Formats a sequence to be printed.
         * @since 0.1.1
         */
        template <>
        struct formatter<sequence> : public formatter<encoder::buffer>
        {};

        /**
         * Formats a sequence slice to be printed.
         * @since 0.1.1
         */
        template <>
        struct formatter<sequence_view> : public formatter<encoder::buffer>
        {};
    }
}
