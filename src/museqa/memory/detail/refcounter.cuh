/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The reference counter for pointers implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2025-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>

#include <museqa/environment.h>
#include <museqa/utility.cuh>

MUSEQA_BEGIN_NAMESPACE

namespace memory::detail
{
    /**
     * A generic reference ownership counter. The reference counter should be composed
     * to relevant objects via inheritance and its methods' visibility should be
     * observed, respected and replicated by its inheritors.
     * @since 1.0
     */
    class refcounter_t
    {
        private:
            intptr_t m_counter = 0;

        private:
            template <typename T>
            using is_refcount_enabled_t = std::enable_if_t<std::is_base_of_v<refcounter_t, T>>;

        protected:
            MUSEQA_CONSTEXPR refcounter_t() noexcept = default;
            MUSEQA_CONSTEXPR refcounter_t(const refcounter_t&) = delete;
            MUSEQA_CONSTEXPR refcounter_t(refcounter_t&&) = delete;

            MUSEQA_INLINE refcounter_t& operator=(const refcounter_t&) = delete;
            MUSEQA_INLINE refcounter_t& operator=(refcounter_t&&) = delete;

            MUSEQA_INLINE virtual ~refcounter_t() = default;

        public:
            /**
             * Creates a new instance of the given reference-counter enabled object
             * and instantly acquires its ownership.
             * @tparam T The reference-counter enabled type to create instance of.
             * @tparam P The types of arguments given to create the new instance.
             * @param args The constructor arguments for the instantiation.
             * @return The owned pointer to the new instance.
             */
            template <typename T, typename ...P, typename = is_refcount_enabled_t<T>>
            MUSEQA_INLINE static auto acquire_ownership(P&&... args) -> T*
            {
                return acquire_ownership(
                    new T(std::forward<decltype(args)>(args)...)
                );
            }

            /**
             * Acquires ownership of a reference-counter enabled object instance.
             * @tparam T The reference-counter enabled type to share ownership of.
             * @param refcounter The instance to acquire ownership of.
             * @return The owned pointer to the given instance.
             */
            template <typename T, typename = is_refcount_enabled_t<T>>
            MUSEQA_CUDA_INLINE static auto acquire_ownership(T *refcounter) noexcept -> T*
            {
              #if MUSEQA_RUNTIME_HOST
                if (refcounter)
                    ++refcounter->m_counter;
              #endif
                return refcounter;
            }

            /**
             * Releases ownership of a reference-counter enabled object instance.
             * @tparam T The reference-counter enabled type to release ownership.
             * @param refcounter The instance to release ownership of.
             */
            template <typename T, typename = is_refcount_enabled_t<T>>
            MUSEQA_CUDA_INLINE static void release_ownership(T *refcounter) MUSEQA_SAFE_EXCEPT
            {
              #if MUSEQA_RUNTIME_HOST
                if (refcounter && --refcounter->m_counter <= 0)
                    delete refcounter;
              #endif
            }
    };
}

MUSEQA_END_NAMESPACE
