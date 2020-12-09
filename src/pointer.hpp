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
             * @since 0.1.1
             */
            struct metadata
            {
                void *ptr = nullptr;                        /// The raw target pointer.
                size_t use_count = 0;                       /// The number of currently active pointer references.

                using allocator_type = museqa::allocator;   /// The pointer's allocator type.
                const allocator_type allocator;             /// The pointer's allocator.

                /**
                 * Initializes a new pointer metadata.
                 * @param ptr The raw pointer to be held.
                 * @param allocator The pointer's allocator.
                 * @see detail::pointer::acquire
                 */
                inline metadata(void *ptr, const allocator_type& allocator) noexcept
                :   ptr {ptr}
                ,   use_count {1}
                ,   allocator {allocator}
                {}

                /**
                 * Deletes and frees the memory claimed by the raw pointer.
                 * @see detail::pointer::release
                 */
                inline ~metadata()
                {
                    allocator.deallocate(ptr);
                }

                /**
                 * Creates a new pointer context from given arguments.
                 * @param ptr The pointer to be put into context.
                 * @param allocator The pointer's allocator.
                 * @return The new pointer's metadata instance.
                 */
                static inline metadata *acquire(void *ptr, const allocator_type& allocator) noexcept
                {
                    return new metadata {ptr, allocator};
                }

                /**
                 * Acquires access to an already existing pointer.
                 * @param meta The metadata of pointer to be acquired.
                 * @return The acquired metadata pointer.
                 */
                __host__ __device__ static inline metadata *acquire(metadata *meta) noexcept
                {
                    if(meta) ++meta->use_count;
                    return meta;
                }

                /**
                 * Releases access to pointer, and deletes it if so needed.
                 * @param meta The metadata of pointer to be released.
                 * @see metadata::acquire
                 */
                __host__ __device__ static inline void release(metadata *meta)
                {
                    #if defined(__museqa_runtime_host)
                        if(meta && --meta->use_count <= 0)
                            delete meta;
                    #endif
                }
            };
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

        public:
            using element_type = pure<T>;               /// The type of pointer's elements.
            using allocator_type = museqa::allocator;   /// The type of allocator for given type.

        protected:
            using metadata = detail::pointer::metadata; /// The internal pointer reference metadata type.

        protected:
            element_type *m_ptr = nullptr;              /// The encapsulated pointer.
            metadata *m_meta = nullptr;                 /// The pointer's metadata.

        public:
            __host__ __device__ constexpr inline pointer() noexcept = default;

            /**
             * Builds a new instance from a raw pointer.
             * @param ptr The pointer to be encapsulated.
             */
            inline pointer(element_type *ptr) noexcept
            :   pointer {ptr, allocator_type::builtin<T>()}
            {}

            /**
             * Builds a new instance from a raw pointer.
             * @param ptr The pointer to be encapsulated.
             * @param allocator The allocator of given pointer.
             */
            inline pointer(element_type *ptr, const allocator_type& allocator) noexcept
            :   pointer {ptr, metadata::acquire(ptr, allocator)}
            {}

            /**
             * Builds a new instance by copying another instance.
             * @param other The instance to be copied.
             */
            __host__ __device__ inline pointer(const pointer& other) noexcept
            :   pointer {other.m_ptr, metadata::acquire(other.m_meta)}
            {}

            /**
             * Gets reference to an already existing pointer.
             * @tparam U The other pointer's type.
             * @param other The reference to be acquired.
             */
            template <typename U>
            __host__ __device__ inline pointer(const pointer<U>& other) noexcept
            :   pointer {static_cast<element_type *>(other.m_ptr), metadata::acquire(other.m_meta)}
            {}

            /**
             * Builds a new instance by moving another instance.
             * @param other The instance to be moved.
             */
            __host__ __device__ inline pointer(pointer&& other) noexcept
            :   pointer {other.m_ptr, metadata::acquire(other.m_meta)}
            {
                other.reset();
            }

            /**
             * Acquires a moved reference to an already existing pointer.
             * @tparam U The other pointer's type.
             * @param other The reference to be moved.
             */
            template <typename U>
            __host__ __device__ inline pointer(pointer<U>&& other) noexcept
            :   pointer {static_cast<element_type *>(other.m_ptr), metadata::acquire(other.m_meta)}
            {
                other.reset();
            }

            /**
             * Releases the acquired pointer reference.
             * @see pointer::pointer
             */
            __host__ __device__ inline ~pointer()
            {
                metadata::release(m_meta);
            }

            /**
             * The copy-assignment operator.
             * @tparam U The other pointer's type.
             * @param other The reference to be acquired.
             * @return This pointer object.
             */
            __host__ __device__ inline pointer& operator=(const pointer& other)
            {
                metadata::release(m_meta);
                return *new (this) pointer {other};
            }

            /**
             * The copy-assignment operator from a different pointer type.
             * @tparam U The other pointer's type.
             * @param other The reference to be acquired.
             * @return This pointer object.
             */
            template <typename U>
            __host__ __device__ inline pointer& operator=(const pointer<U>& other)
            {
                metadata::release(m_meta);
                return *new (this) pointer {other};
            }

            /**
             * The move-assignment operator.
             * @tparam U The other pointer's type.
             * @param other The reference to be acquired.
             * @return This pointer object.
             */
            __host__ __device__ inline pointer& operator=(pointer&& other)
            {
                metadata::release(m_meta);
                return *new (this) pointer {std::forward<decltype(other)>(other)};
            }

            /**
             * The move-assignment operator from a different pointer type.
             * @tparam U The other pointer's type.
             * @param other The reference to be acquired.
             * @return This pointer object.
             */
            template <typename U>
            __host__ __device__ inline pointer& operator=(pointer<U>&& other)
            {
                metadata::release(m_meta);
                return *new (this) pointer {std::forward<decltype(other)>(other)};
            }

            /**
             * Dereferences the pointer.
             * @return The pointed object.
             */
            template <typename Z = T>
            __host__ __device__ inline auto operator*() noexcept
            -> typename std::enable_if<!std::is_void<Z>::value, pure<Z>&>::type
            {
                return *m_ptr;
            }

            /**
             * Dereferences the const-qualified pointer.
             * @return The const-qualified pointed object.
             */
            template <typename Z = T>
            __host__ __device__ inline auto operator*() const noexcept
            -> typename std::enable_if<!std::is_void<Z>::value, const pure<Z>&>::type
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
             * Gives access to the raw const-qualified pointer.
             * @return The raw const-qualified pointer.
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
             * Gives access to raw const-qualified pointer using the dereference operator.
             * @return The raw const-qualified pointer.
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
            template <typename Z = T>
            __host__ __device__ inline auto operator[](ptrdiff_t offset) noexcept
            -> typename std::enable_if<!std::is_void<Z>::value, pure<Z>&>::type
            {
                static_assert(std::is_array<T>::value, "only array pointers have valid offsets");
                return m_ptr[offset];
            }

            /**
             * Gives access to an const-qualified object in an array pointer offset.
             * @param offset The offset to be accessed.
             * @return The requested const-qualified object instance.
             */
            template <typename Z = T>
            __host__ __device__ inline auto operator[](ptrdiff_t offset) const noexcept
            -> typename std::enable_if<!std::is_void<Z>::value, const pure<Z>&>::type
            {
                static_assert(std::is_array<T>::value, "only array pointers have valid offsets");
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
             * Converts to raw const-qualified pointer type.
             * @return The const-qualified pointer converted to raw type.
             */
            __host__ __device__ inline operator const element_type *() const noexcept
            {
                return m_ptr;
            }

            /**
             * Gives access to the raw const-qualified pointer.
             * @return The raw const-qualified pointer.
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
                static_assert(std::is_array<T>::value, "only array pointers have valid offsets");
                return pointer {m_ptr + offset, metadata::acquire(m_meta)};
            }

            /**
             * Resets the pointer manager to an empty state.
             * @see pointer::pointer
             */
            __host__ __device__ inline void reset() noexcept
            {
                metadata::release(m_meta);
                new (this) pointer {};
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
                return m_meta ? m_meta->use_count : 0;
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
             * @tparam U The raw pointer's type.
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
            __host__ __device__ inline pointer(element_type *ptr, metadata *meta) noexcept
            :   m_ptr {ptr}
            ,   m_meta {meta}
            {}

        template <typename> friend class pointer;
    };
}
