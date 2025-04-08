/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements a pointer metadata storage and instance counter.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

#include <cstddef>
#include <utility>

#include <museqa/environment.h>
#include <museqa/utility.cuh>

#include <museqa/memory/allocator.cuh>
#include <museqa/memory/pointer/wrapper.cuh>
#include <museqa/memory/detail/refcounter.cuh>

MUSEQA_BEGIN_NAMESPACE

namespace memory::pointer::detail
{
    /**
     * The context metadata and reference counter for a smart pointer.
     * @tparam A The allocator type for the target instance pointer.
     * @since 1.0
     */
    template <typename A>
    class metadata_t : public memory::detail::refcounter_t
    {
        private:
            using wrapper_t = memory::pointer::wrapper_t<void>;
            using allocator_t = A;

        static_assert(std::is_base_of_v<memory::detail::deleter_t, A>
          , "composed metadata type must inherit at least from a deleter type");

        private:
            wrapper_t m_ptr;
            allocator_t m_allocator;

        public:
            /**
             * Creates a new pointer context metadata from a raw pointer instance.
             * @param ptr The raw pointer wrapper to acquire ownership of.
             * @param allocator The allocator instance for the wrapped pointer.
             */
            MUSEQA_CUDA_INLINE metadata_t(wrapper_t ptr, const allocator_t& allocator)
              : m_ptr (ptr)
              , m_allocator (allocator)
            {}

            /**
             * Releases ownership and frees the wrapped pointer's memory region.
             * @see museqa::memory::pointer::detail::metadata_t::metadata_t
             */
            MUSEQA_CUDA_INLINE ~metadata_t() override
            {
                if (!m_ptr.empty()) {
                    m_allocator.deallocate(m_ptr.unwrap(), 0);
                }
            }
    };
}

MUSEQA_END_NAMESPACE
