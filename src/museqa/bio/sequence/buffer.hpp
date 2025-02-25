/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The representation of a biological sequence buffer.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2023-present Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <cstdint>

#include <museqa/environment.h>
#include <museqa/utility.hpp>
#include <museqa/bio/sequence/block.hpp>
#include <museqa/memory/allocator.hpp>
#include <museqa/memory/pointer/shared.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace bio::sequence
{
    /**
     * The buffer type for a biological sequence. Internally, sequence symbols are
     * stored compressed within blocks that are easily accessible and decompressible.
     * @since 1.0
     */
    class buffer_t : protected memory::pointer::shared_t<block_t>
    {
        private:
            typedef memory::pointer::shared_t<block_t> underlying_t;

        protected:
            size_t m_length = 0;

        public:
            MUSEQA_INLINE buffer_t() noexcept = default;
            MUSEQA_INLINE buffer_t(const buffer_t&) MUSEQA_SAFE_EXCEPT = default;
            MUSEQA_INLINE buffer_t(buffer_t&&) MUSEQA_SAFE_EXCEPT = default;

            MUSEQA_INLINE buffer_t& operator=(const buffer_t&) MUSEQA_SAFE_EXCEPT = default;
            MUSEQA_INLINE buffer_t& operator=(buffer_t&&) MUSEQA_SAFE_EXCEPT = default;

            /**
             * Informs the number of symbols within the current sequence buffer.
             * @return The length of the sequence.
             */
            MUSEQA_CUDA_INLINE size_t length() const noexcept
            {
                return m_length;
            }

            /**
             * Informs whether the sequence buffer is empty and has zero elements.
             * @return Is the sequence buffer empty?
             */
            MUSEQA_CUDA_INLINE bool empty() const noexcept
            {
                return m_length == 0 || underlying_t::empty();
            }

        protected:
            /**
             * Initializes a new sequence buffer from an existing buffer pointer.
             * @param ptr The buffer pointer to initialize sequence buffer from.
             * @param length The number of symbols contained by the sequence.
             */
            MUSEQA_CUDA_INLINE buffer_t(const underlying_t& ptr, size_t length) MUSEQA_SAFE_EXCEPT
              : underlying_t (ptr)
              , m_length (length)
            {}

        friend buffer_t encode(const char*, const size_t, const memory::allocator_t&);
        friend buffer_t encode(const std::string&, const memory::allocator_t&);
        friend std::string decode(const buffer_t&);
    };
}

MUSEQA_END_NAMESPACE
