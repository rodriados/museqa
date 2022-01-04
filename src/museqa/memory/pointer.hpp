/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Exposes pointer wrapper types and helper pointer functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>
#include <cstring>
#include <utility>

#include <museqa/environment.h>

#include <museqa/memory/pointer/shared.hpp>
#include <museqa/memory/pointer/unique.hpp>
#include <museqa/memory/pointer/unmanaged.hpp>
#include <museqa/memory/pointer/wrapper.hpp>
#include <museqa/memory/pointer/exception.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace memory
{
    /**
     * Copies data from a source to a target pointer.
     * @tparam T The pointers' content type.
     * @param target The pointer to copy data into.
     * @param source The pointer to copy data from.
     * @param count The number of elements to be copied.
     */
    template <typename T = void>
    inline void copy(T *target, const T *source, size_t count = 1) noexcept
    {
        using U = typename std::conditional<std::is_void<T>::value, uint8_t, T>::type;
        std::memmove(target, source, count * sizeof(U));
    }

    /**
     * Initializes a memory region with zeroes.
     * @tparam T The pointer's content type.
     * @param target The memory region pointer to be zero-initialized.
     * @param count The number of elements to be initialized.
     */
    template <typename T = void>
    inline void zero(T *target, size_t count = 1) noexcept
    {
        using U = typename std::conditional<std::is_void<T>::value, uint8_t, T>::type;
        std::memset(target, 0, count * sizeof(U));
    }
}

MUSEQA_END_NAMESPACE
