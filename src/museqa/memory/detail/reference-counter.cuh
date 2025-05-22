/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The reference counter for pointers implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2025-present Rodrigo Siqueira
 */
#pragma once

#include <atomic>
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
    class reference_counter_t
    {
        private:
          #ifndef MUSEQA_REFERENCE_COUNTER_AVOID_ATOMIC
            std::atomic_intptr_t m_counter = 0;
          #else
            intptr_t m_counter = 0;
          #endif

        protected:
            MUSEQA_CONSTEXPR reference_counter_t() noexcept = default;
            MUSEQA_CONSTEXPR reference_counter_t(const reference_counter_t&) = delete;
            MUSEQA_CONSTEXPR reference_counter_t(reference_counter_t&&) = delete;

            MUSEQA_INLINE reference_counter_t& operator=(const reference_counter_t&) = delete;
            MUSEQA_INLINE reference_counter_t& operator=(reference_counter_t&&) = delete;

            MUSEQA_INLINE virtual ~reference_counter_t() {}

        template <typename T> friend MUSEQA_CUDA_INLINE auto share_ownership(T*) noexcept -> T*;
        template <typename T> friend MUSEQA_CUDA_INLINE void release_ownership(T*);
    };

    /**
     * Indicates whether the given type is a reference-counter.
     * @tparam T The type to check if it is a reference-counter.
     * @since 1.0
     */
    template <typename T>
    MUSEQA_CONSTEXPR bool is_reference_counter = std::is_base_of_v<reference_counter_t, T>;

    /**
     * Creates a new instance of the given reference-counter-enabled type, using
     * the given parameters, and instantly acquires ownership of it.
     * @tparam T The reference-counter enabled type to create a new instance of.
     * @tparam P The types of arguments given to create the new instance.
     * @param args The constructor arguments for the type instantiation.
     * @return The acquired pointer to the new type instance.
     */
    template <typename T, typename ...P>
    MUSEQA_INLINE auto acquire_ownership(P&&... args) -> T*
    {
        static_assert(is_reference_counter<T>
          , "cannot acquire ownership of type that is not a reference-counter");
        return share_ownership(new T(std::forward<decltype(args)>(args)...));
    }

    /**
     * Acquires shared ownership of an instance of a reference-counter-enabled type.
     * @tparam T The reference-counter-enabled type to share ownership of.
     * @param refcounter The instance to acquire shared ownership of.
     * @return The shared pointer to the given type instance.
     */
    template <typename T>
    MUSEQA_CUDA_INLINE auto share_ownership(T *refcounter) noexcept -> T*
    {
        static_assert(is_reference_counter<T>
          , "cannot acquire ownership of type that is not a reference-counter");
      #if MUSEQA_RUNTIME_HOST
        if (refcounter)
            ++refcounter->m_counter;
      #endif
        return refcounter;
    }

    /**
     * Releases ownership of an instance of a reference-counter-enabled type.
     * @tparam T The reference-counter-enabled type to release ownership.
     * @param refcounter The instance to release ownership of.
     */
    template <typename T>
    MUSEQA_CUDA_INLINE void release_ownership(T *refcounter)
    {
        static_assert(is_reference_counter<T>
          , "cannot release ownership of type that is not a reference-counter");
      #if MUSEQA_RUNTIME_HOST
        if (refcounter && --refcounter->m_counter <= 0)
            delete refcounter;
      #endif
    }
}

MUSEQA_END_NAMESPACE
