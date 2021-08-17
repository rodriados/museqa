/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Imports the whole codebase for memory pointers.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>
#include <cstring>
#include <utility>

#include <museqa/utility.hpp>

#include <museqa/memory/pointer/common.hpp>
#include <museqa/memory/pointer/shared.hpp>
#include <museqa/memory/pointer/unique.hpp>
#include <museqa/memory/pointer/weak.hpp>

namespace museqa
{
    namespace memory
    {
        /**
         * Copies data from the source to the target pointer.
         * @tparam T The pointer's contents type.
         * @param target The pointer to copy data into.
         * @param source The pointer to copy data from.
         * @param count The total number of elements to copy.
         */
        template <typename T>
        inline void copy(T *target, const T *source, size_t count = 1) noexcept
        {
            constexpr size_t size = sizeof(typename std::conditional<std::is_void<T>::value, char, T>::type);
            std::memcpy(target, source, size * count);
        }

        /**
         * Initializes a memory region to the given byte value.
         * @tparam T The pointer's contents type.
         * @param target The memory region pointer to be initialized.
         * @param byte The byte value to initialize the memory region with.
         * @param count The total number of elements to be initialized.
         */
        template <typename T>
        inline void set(T *target, uint8_t byte, size_t count = 1) noexcept
        {
            constexpr size_t size = sizeof(typename std::conditional<std::is_void<T>::value, char, T>::type);
            std::memset(target, byte, size * count);
        }

        /**
         * Initializes a memory region to zero.
         * @tparam T The pointer's contents type.
         * @param target The memory region pointer to be initialized.
         * @param count The total number of elements to be zero-initialized.
         */
        template <typename T>
        inline void zero(T *target, size_t count = 1) noexcept
        {
            memory::set(target, (uint8_t) 0, count);
        }
    }
}
