/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements generic CUDA-compatible smart pointers.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <utility>

#include "utils.hpp"
#include "allocator.hpp"

namespace museqa
{
    namespace detail
    {
        namespace pointer
        {
            /**
             * Keeps track of the number of references to a given memory pointer.
             * @tparam T The pointer type.
             * @since 0.1.1
             */
            template <typename T>
            struct counter
            {
                using element_type = T;                     /// The type of elements represented by the pointer.
                using allocator_type = museqa::allocator;   /// The type of allocator for given type.

                element_type *ptr = nullptr;                /// The raw target pointer.
                size_t use_count = 0;                       /// The number of currently active pointer references.

                const allocator_type allocator;             /// The pointer's allocator.

                /**
                 * Initializes a new pointer counter.
                 * @param ptr The raw pointer to be held.
                 * @param allocator The pointer's allocator.
                 * @see detail::pointer::acquire
                 */
                inline counter(element_type *ptr, const allocator_type& allocator) noexcept
                :   ptr {ptr}
                ,   use_count {1}
                ,   allocator {allocator}
                {}

                /**
                 * Deletes and frees the memory claimed by the raw pointer.
                 * @see detail::pointer::release
                 */
                inline ~counter()
                {
                    allocator.deallocate(ptr);
                }
            };

            /**
             * Creates a new pointer context from given arguments.
             * @tparam T The pointer type.
             * @param ptr The pointer to be put into context.
             * @param allocator The pointer's allocator.
             * @return The new pointer's counter instance.
             */
            template <typename T>
            inline counter<T> *acquire(T *ptr, const museqa::allocator& allocator) noexcept
            {
                return new counter<T> {ptr, allocator};
            }

            /**
             * Acquires access to an already existing pointer.
             * @tparam T The pointer type.
             * @param meta The metadata of pointer to be acquired.
             * @return The metadata acquired pointer.
             */
            template <typename T>
            __host__ __device__ inline counter<T> *acquire(counter<T> *meta) noexcept
            {
                meta && ++meta->use_count;
                return meta;
            }

            /**
             * Releases access to pointer, and deletes it if so needed.
             * @tparam T The pointer type.
             * @param meta The metadata of pointer to be released.
             * @see detail::pointer::acquire
             */
            template <typename T>
            __host__ __device__ inline void release(counter<T> *meta)
            {
                #if defined(__museqa_runtime_host)
                    if(meta && --meta->use_count <= 0)
                        delete meta;
                #endif
            }
        }
    }

    /**
     * Implements a smart pointer. This class can be used to represent a pointer
     * that is managed and deleted automatically when all references to it have
     * been destroyed.
     * @tparam T The type of pointer to be held.
     * @since 0.1.1
     */
    template <typename T>
    class pointer
    {
        static_assert(!std::is_function<T>::value, "cannot create pointer to a function");
        static_assert(!std::is_reference<T>::value, "cannot create pointer to a reference");
        static_assert(!std::is_same<T, void>::value, "cannot create void smart pointers");

        public:
            using element_type = pure<T>;               /// The type of pointer's elements.
            using allocator_type = museqa::allocator;   /// The type of allocator for given type.

        protected:
            using counter_type = detail::pointer::counter<element_type>;    /// The pointer counter type.

        protected:
            element_type *m_ptr = nullptr;              /// The encapsulated pointer.
            counter_type *m_meta = nullptr;             /// The pointer's metadata.

        public:
            __host__ __device__ constexpr inline pointer() noexcept = default;

            /**
             * Builds a new instance from a raw pointer.
             * @param ptr The pointer to be encapsulated.
             */
            inline pointer(element_type *ptr) noexcept
            :   pointer {ptr, detail::pointer::acquire(ptr, allocator_type::builtin<T>())}
            {}

            /**
             * Builds a new instance from a raw pointer.
             * @param ptr The pointer to be encapsulated.
             * @param allocator The allocator of given pointer.
             */
            inline pointer(element_type *ptr, const allocator_type& allocator) noexcept
            :   pointer {ptr, detail::pointer::acquire(ptr, allocator)}
            {}

            /**
             * Gets reference to an already existing pointer.
             * @param other The reference to be acquired.
             */
            __host__ __device__ inline pointer(const pointer& other) noexcept
            :   pointer {other.m_ptr, detail::pointer::acquire(other.m_meta)}
            {}

            /**
             * Acquires a moved reference to an already existing pointer.
             * @param other The reference to be moved.
             */
            __host__ __device__ inline pointer(pointer&& other) noexcept
            {
                operator=(std::forward<decltype(other)>(other));
            }

            /**
             * Releases the acquired pointer reference.
             * @see pointer::pointer
             */
            __host__ __device__ inline ~pointer()
            {
                detail::pointer::release(m_meta);
            }

            /**
             * The copy-assignment operator.
             * @param other The reference to be acquired.
             * @return This pointer object.
             */
            __host__ __device__ inline pointer& operator=(const pointer& other)
            {
                detail::pointer::release(m_meta);
                m_meta = detail::pointer::acquire(other.m_meta);
                m_ptr = other.m_ptr;
                return *this;
            }

            /**
             * The move-assignment operator.
             * @param other The reference to be acquired.
             * @return This pointer object.
             */
            __host__ __device__ inline pointer& operator=(pointer&& other)
            {
                other.swap(*this);
                other.reset();
                return *this;
            }

            /**
             * Dereferences the pointer.
             * @return The pointed object.
             */
            __host__ __device__ inline element_type& operator*() noexcept
            {
                return *m_ptr;
            }

            /**
             * Dereferences the constant pointer.
             * @return The constant pointed object.
             */
            __host__ __device__ inline const element_type& operator*() const noexcept
            {
                return *m_ptr;
            }

            /**
             * Gives access to the raw pointer.
             * @return The raw pointer.
             */
            __host__ __device__ inline element_type *operator&() noexcept
            {
                return m_ptr;
            }

            /**
             * Gives access to the raw constant pointer.
             * @return The raw constant pointer.
             */
            __host__ __device__ inline const element_type *operator&() const noexcept
            {
                return m_ptr;
            }

            /**
             * Gives access to raw pointer using the dereference operator.
             * @return The raw pointer.
             */
            __host__ __device__ inline element_type *operator->() noexcept
            {
                return m_ptr;
            }

            /**
             * Gives access to raw constant pointer using the dereference operator.
             * @return The raw constant pointer.
             */
            __host__ __device__ inline const element_type *operator->() const noexcept
            {
                return m_ptr;
            }

            /**
             * Gives access to an object in an array pointer offset.
             * @param offset The offset to be accessed.
             * @return The requested object instance.
             */
            __host__ __device__ inline element_type& operator[](ptrdiff_t offset) noexcept
            {
                static_assert(!std::is_same<element_type, T>::value, "only array pointers have valid offsets");
                return m_ptr[offset];
            }

            /**
             * Gives access to an constant object in an array pointer offset.
             * @param offset The offset to be accessed.
             * @return The requested constant object instance.
             */
            __host__ __device__ inline const element_type& operator[](ptrdiff_t offset) const noexcept
            {
                static_assert(!std::is_same<element_type, T>::value, "only array pointers hava valid offsets");
                return m_ptr[offset];
            }

            /**
             * Checks if the stored pointer is not null.
             * @return Is the pointer not null?
             */
            __host__ __device__ inline operator bool() const noexcept
            {
                return (m_ptr != nullptr);
            }

            /**
             * Converts to raw pointer type.
             * @return The pointer converted to raw type.
             */
            __host__ __device__ inline operator element_type *() noexcept
            {
                return m_ptr;
            }

            /**
             * Converts to raw constant pointer type.
             * @return The constant pointer converted to raw type.
             */
            __host__ __device__ inline operator const element_type *() const noexcept
            {
                return m_ptr;
            }

            /**
             * Gives access to raw pointer.
             * @return The raw pointer.
             */
            __host__ __device__ inline const element_type *get() const noexcept
            {
                return m_ptr;
            }

            /**
             * Gets an instance to an offset pointer.
             * @param offset The requested offset.
             * @return The new offset pointer instance.
             */
            __host__ __device__ inline pointer offset(ptrdiff_t offset) noexcept
            {
                static_assert(!std::is_same<element_type, T>::value, "only array pointers have valid offsets");
                return pointer {m_ptr + offset, detail::pointer::acquire(m_meta)};
            }

            /**
             * Resets the pointer manager to an empty state.
             * @see pointer::pointer
             */
            __host__ __device__ inline void reset() noexcept
            {
                detail::pointer::release(m_meta);
                m_ptr = nullptr;
                m_meta = nullptr;
            }

            /**
             * Swaps the contents with another pointer instance.
             * @param other The target instance to swap with.
             */
            __host__ __device__ inline void swap(pointer& other) noexcept
            {
                utils::swap(m_ptr, other.m_ptr);
                utils::swap(m_meta, other.m_meta);
            }

            /**
             * Returns the reference to the current pointer's allocator.
             * @return The pointer's allocator instance.
             */
            inline allocator_type allocator() const noexcept
            {
                return m_meta ? m_meta->allocator : allocator_type::builtin<T>();
            }

            /**
             * Informs the number of references created to pointer.
             * @return The number of references to pointer.
             */
            inline size_t use_count() const noexcept
            {
                return m_meta ? m_meta->count : 0;
            }

            /**
             * Allocates a new array pointer of given size.
             * @param count The number of elements to be allocated.
             * @return The newly allocated pointer.
             */
            static inline auto make(size_t count = 1) noexcept -> pointer
            {
                return make(allocator_type::builtin<T>(), count);
            }

            /**
             * Allocates a new array pointer of given size with an allocator.
             * @param allocator The allocator to be used to new pointer.
             * @param count The number of elements to be allocated.
             * @return The newly allocated pointer.
             */
            static inline auto make(const allocator_type& allocator, size_t count = 1) noexcept -> pointer
            {
                element_type *ptr = allocator.allocate<element_type>(count);
                return pointer {ptr, allocator};
            }

            /**
             * Creates a non-owning weak reference to given pointer.
             * @param ptr The non-owning pointer to create weak-reference from.
             * @return The weak referenced pointer instance.
             */
            __host__ __device__ static inline auto weak(element_type *ptr) noexcept -> pointer
            {
                return pointer {ptr, nullptr};
            }

        protected:
            /**
             * Builds a new instance from custom internal parts.
             * @param ptr The raw pointer object.
             * @param meta The pointer's metadata.
             */
            __host__ __device__ inline pointer(element_type *ptr, counter_type *meta) noexcept
            :   m_ptr {ptr}
            ,   m_meta {meta}
            {}
    };
}
