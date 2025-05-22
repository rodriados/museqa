/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements a pointer control-block and reference counter.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2025-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>
#include <museqa/utility.cuh>

#include <museqa/memory/allocator.hpp>
#include <museqa/memory/pointer/wrapper.cuh>
#include <museqa/memory/detail/reference-counter.cuh>

MUSEQA_BEGIN_NAMESPACE

namespace memory::detail
{
    /**
     * The control block and reference counter for a smart pointer.
     * @tparam A The allocator type for the target instance pointer.
     * @since 1.0
     */
    template <
        typename A
      , typename = std::enable_if_t<is_allocator<A>>>
    class control_block_t : public reference_counter_t
    {
        private:
            using wrapper_t = pointer::wrapper_t<void>;
            using allocator_t = A;

        private:
            wrapper_t m_ptr;
            allocator_t m_allocator;

        public:
            /**
             * Creates a new pointer context metadata from a raw pointer instance.
             * @param ptr The raw pointer wrapper to acquire ownership of.
             * @param allocator The allocator instance for the wrapped pointer.
             */
            MUSEQA_INLINE control_block_t(wrapper_t ptr, const allocator_t& allocator)
              : m_ptr (ptr)
              , m_allocator (allocator)
            {}

            /**
             * Releases ownership and frees the wrapped pointer's memory region.
             * @see museqa::memory::pointer::detail::control_block_t::control_block_t
             */
            MUSEQA_INLINE ~control_block_t() override
            {
                if (!m_ptr.empty()) {
                    m_allocator.deallocate(m_ptr.unwrap(), 0);
                }
            }
    };
}

MUSEQA_END_NAMESPACE
