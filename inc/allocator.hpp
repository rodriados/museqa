/** 
 * Multiple Sequence Alignment allocator header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019 Rodrigo Siqueira
 */
#pragma once

#include <utils.hpp>

namespace msa
{
    /**
     * Describes the allocation and deallocation routines for a given type.
     * @tparam T The pointer's element type.
     * @since 0.1.1
     */
    class allocator
    {
        public:
            using ptr_type = void *;                            /// The type of pointer to allocate.
            using up_type = functor<void(ptr_type *, size_t)>;  /// The allocator's functor type.
            using down_type = functor<void(ptr_type)>;          /// The deallocator's functor type.

        protected:
            up_type m_up;                       /// The allocator's up functor.
            down_type m_down;                   /// The allocator's down functor.

        public:
            inline constexpr allocator() noexcept = delete;
            inline constexpr allocator(const allocator&) noexcept = default;
            inline constexpr allocator(allocator&&) noexcept = default;

            /**
             * Instantiates a new allocator with the given functors.
             * @param up_functor The allocator functor.
             * @param down_functor The deallocator functor.
             */
            inline constexpr allocator(up_type up_functor, down_type down_functor) noexcept
            :   m_up {up_functor}
            ,   m_down {down_functor}
            {}

            /**
             * Instantiates a new allocator from given lambdas.
             * @tparam A The allocator functor lambda type.
             * @tparam D The deallocator functor lambda type.
             * @param up_lambda The allocator lambda.
             * @param down_lambda The deallocator lambda.
             */
            template <typename A, typename D>
            inline constexpr allocator(A up_lambda, D down_lambda) noexcept
            :   allocator {up_type(up_lambda), down_type(down_lambda)}
            {}

            inline allocator& operator=(const allocator&) noexcept = default;
            inline allocator& operator=(allocator&&) noexcept = default;

            /**
             * Invokes the allocator functor and creates a new allocated pointer.
             * @param ptr The target pointer to allocated memory to.
             * @param count The number of elements to allocate memory to.
             * @return The newly allocated pointer.
             */
            inline auto allocate(ptr_type *ptr, size_t count) const -> ptr_type
            {
                (m_up)(ptr, count);
                return *ptr;
            }

            /**
             * Invokes the allocator functor and creates a new allocated pointer.
             * @param count The number of elements to allocate memory to.
             * @return The newly allocated pointer.
             */
            inline auto allocate(size_t count) const -> ptr_type
            {
                ptr_type ptr;
                return allocate(&ptr, count);
            }

            /**
             * Invokes the deallocator functor and frees the pointer's memory.
             * @param ptr The pointer of which memory must be freed.
             */
            inline void deallocate(ptr_type ptr) const
            {
                (m_down)(ptr);
            }

            /**
             * Creates a builtin allocator for a specified pointer type, which
             * has its default contructor called for each instance.
             * @tparam T The type of pointer element to build.
             * @return The new allocator for given type.
             */
            template <typename T>
            inline static auto builtin() -> allocator
            {
                return {
                    [](void **ptr, size_t count) { *ptr = new T [count]; }
                ,   [](void *ptr) { delete[] (static_cast<T*>(ptr)); }
                };
            }
    };
}
