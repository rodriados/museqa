/** 
 * Multiple Sequence Alignment allocator header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019 Rodrigo Siqueira
 */
#pragma once

#include <utils.hpp>

namespace msa
{
    namespace detail
    {
        namespace memory
        {
            /**
             * The default pointer allocator.
             * @tparam T The pointer's content type.
             * @param ptr Output parameter for the newly allocated pointer.
             * @param count The number of elements to allocate.
             */
            template <typename T>
            inline void up(T **ptr, size_t count)
            {
                *ptr = new T[count];
            }

            /**
             * The default pointer deallocator.
             * @tparam T The pointer's content type.
             * @param ptr The pointer to be deallocated.
             */
            template <typename T>
            inline void down(T *ptr)
            {
                delete[] ptr;
            }
        }

        /**
         * Describes the allocation and deallocation routines for a given type.
         * @tparam T The pointer's element type.
         * @since 0.1.1
         */
        template <typename T>
        class allocator
        {
            public:
                using element_type = T;                             /// The allocator's element type.
                using ptr_type = element_type *;                    /// The type of pointer to allocate.
                using up_type = functor<void(ptr_type *, size_t)>;  /// The allocator's functor type.
                using down_type = functor<void(ptr_type)>;          /// The deallocator's functor type.

            protected:
                up_type m_up = memory::up<T>;       /// The allocator's up functor.
                down_type m_down = memory::down<T>; /// The allocator's down functor.

            public:
                inline constexpr allocator() noexcept = default;
                inline constexpr allocator(const allocator&) noexcept = default;
                inline constexpr allocator(allocator&&) noexcept = default;

                /**
                 * Instantiates a new allocator with the given functors.
                 * @param fup The allocator functor.
                 * @param fdown The deallocator functor.
                 */
                inline constexpr allocator(up_type fup, down_type fdown) noexcept
                :   m_up {fup}
                ,   m_down {fdown}
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
        };
    }

    /**
     * Allocates and deallocates pointers of the given type.
     * @tparam T The pointers' element type.
     * @since 0.1.1
     */
    template <typename T>
    using allocator = detail::allocator<pure<T>>;
}
