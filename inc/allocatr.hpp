/** 
 * Multiple Sequence Alignment allocator header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019 Rodrigo Siqueira
 */
#pragma once

#ifndef ALLOCATR_HPP_INCLUDED
#define ALLOCATR_HPP_INCLUDED

#include <utils.hpp>

namespace internal
{
    namespace allocatr
    {
        /**
         * The type of pointer allocator functions.
         * @tparam T The pointer's element type.
         * @since 0.1.1
         */
        template <typename T>
        using up = functor<pure<T> *(size_t)>;

        /**
         * The type of pointer delocator functions.
         * @tparam T The pointer's element type.
         * @since 0.1.1
         */
        template <typename T>
        using down = functor<void(pure<T> *)>;
    }

    /**
     * The default pointer allocator.
     * @tparam T The pointer's element type.
     * @param (ignored) The number of elements to allocate.
     * @return The newly allocated pointer.
     */
    template <typename T>
    inline auto malloc(size_t)
    -> typename std::enable_if<!std::is_array<T>::value, pure<T> *>::type
    {
        using element_type = pure<T>;
        return new element_type;
    }

    /**
     * The default array pointer allocator.
     * @tparam T The pointer's element type.
     * @param count The number of elements to allocate.
     * @return The newly allocated pointer.
     */
    template <typename T>
    inline auto malloc(size_t count)
    -> typename std::enable_if<std::is_array<T>::value, pure<T> *>::type
    {
        using element_type = pure<T>;
        return new element_type [count];
    }

    /**
     * The default pointer deleter.
     * @tparam T The pointer's element type.
     * @param ptr The pointer to be deleted.
     */
    template <typename T>
    inline auto mdeloc(pure<T> *ptr)
    -> typename std::enable_if<!std::is_array<T>::value, void>::type
    {
        delete ptr;
    }

    /**
     * The default array pointer deleter.
     * @tparam T The pointer's element type.
     * @param ptr The pointer to be deleted.
     */
    template <typename T>
    inline auto mdeloc(pure<T> *ptr)
    -> typename std::enable_if<std::is_array<T>::value, void>::type
    {
        delete[] ptr;
    }
}

/**
 * Describes the allocation and deallocation functions for a given type.
 * @tparam T The pointer's element type.
 * @since 0.1.1
 */
template <typename T>
class allocatr
{
    public:
        using up = internal::allocatr::up<T>;       /// The allocator's functor type.
        using down = internal::allocatr::down<T>;   /// The delocator's functor type.
        using element_type = pure<T>;               /// The type of elements to allocate.

    protected:
        up fup = internal::malloc<T>;               /// The pointer's allocator.
        down fdown = internal::mdeloc<T>;           /// The pointer's delocator.

    public:
        inline constexpr allocatr() noexcept = default;
        inline constexpr allocatr(const allocatr&) noexcept = default;
        inline constexpr allocatr(allocatr&&) noexcept = default;

        /**
         * Instantiates a new allocator with the given functors.
         * @param malloc The allocator functor.
         * @param mdeloc The delocator functor.
         */
        inline constexpr allocatr(up malloc, down mdeloc) noexcept
        :   fup {malloc}
        ,   fdown {mdeloc}
        {}

        inline allocatr& operator=(const allocatr&) noexcept = default;
        inline allocatr& operator=(allocatr&&) noexcept = default;

        /**
         * Invokes the allocator functor and returns the newly allocated pointer.
         * @param count The number of elements to allocate memory to.
         * @return The newly allocated pointer.
         */
        inline auto allocate(size_t count) const -> element_type *
        {
            return (fup)(count);
        }

        /**
         * Invokes the delocator functor and frees the pointer's memory.
         * @param ptr The pointer which memory must be freed.
         */
        inline auto delocate(element_type *ptr) const -> void
        {
            (fdown)(ptr);
        }
};

#endif