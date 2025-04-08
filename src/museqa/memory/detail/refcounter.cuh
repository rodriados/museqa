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

        protected:
            MUSEQA_CONSTEXPR refcounter_t() noexcept = default;
            MUSEQA_CONSTEXPR refcounter_t(const refcounter_t&) = delete;
            MUSEQA_CONSTEXPR refcounter_t(refcounter_t&&) = delete;

            MUSEQA_INLINE refcounter_t& operator=(const refcounter_t&) = delete;
            MUSEQA_INLINE refcounter_t& operator=(refcounter_t&&) = delete;

            MUSEQA_CUDA_INLINE virtual ~refcounter_t() {}

        template <typename T> friend MUSEQA_CUDA_ENABLED auto share_ownership(T*) noexcept -> T*;
        template <typename T> friend MUSEQA_CUDA_ENABLED void release_ownership(T*) MUSEQA_SAFE_EXCEPT;
    };

    /**
     * Indicates whether the given type is reference-counter-enabled.
     * @tparam T The type check if reference-counter-enabled.
     * @since 1.0
     */
    template <typename T>
    MUSEQA_CONSTEXPR bool is_refcounter_enabled = std::is_base_of_v<refcounter_t, T>;

    /**
     * Creates a new instance of the given reference-counter-enabled type, using
     * the given parameters, and instantly acquires ownership of it.
     * @tparam T The reference-counter enabled type to create a new instance of.
     * @tparam P The types of arguments given to create the new instance.
     * @param args The constructor arguments for the type instantiation.
     * @return The acquired pointer to the new type instance.
     */
    template <typename T, typename ...P>
    MUSEQA_CUDA_INLINE auto acquire_ownership(P&&... args) -> T*
    {
        static_assert(is_refcounter_enabled<T>
          , "cannot acquire ownership of type that is not reference-counter-enabled");
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
        static_assert(is_refcounter_enabled<T>
          , "cannot acquire ownership of type that is not reference-counter-enabled");
        if (refcounter)
            ++refcounter->m_counter;
        return refcounter;
    }

    /**
     * Releases ownership of an instance of a reference-counter-enabled type.
     * @tparam T The reference-counter-enabled type to release ownership.
     * @param refcounter The instance to release ownership of.
     */
    template <typename T>
    MUSEQA_CUDA_INLINE void release_ownership(T *refcounter) MUSEQA_SAFE_EXCEPT
    {
        static_assert(is_refcounter_enabled<T>
          , "cannot release ownership of type that is not reference-counter-enabled");
        if (refcounter && --refcounter->m_counter <= 0)
            delete refcounter;
    }
}

MUSEQA_END_NAMESPACE
