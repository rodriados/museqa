/** 
 * Multiple Sequence Alignment utilities header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Alexandr Poltavsky, Antony Polukhin, Rodrigo Siqueira
 */
#ifndef UTILS_HPP_INCLUDED
#define UTILS_HPP_INCLUDED

#pragma once

#include <cstddef>
#include <utility>

namespace utils
{
    /**
     * A memory aligned storage container.
     * @tparam S The number of elements in storage.
     * @tparam A The alignment the storage should use.
     * @since 0.1.1
     */
    template <size_t S, size_t A>
    struct AlignedStorage
    {
        alignas(A) char storage[S]; /// The aligned storage container.
    };

    /**
     * Checks whether two types can be compared to each other.
     * @tparam A First type to check.
     * @tparam B Second type to check.
     * @since 0.1.1
     */
    template <typename A, typename B>
    struct Comparable {
        static constexpr bool value = std::is_same<A, B>::value
            || (std::is_arithmetic<A>::value && std::is_arithmetic<B>::value);
    };

    /**
     * Removes a type's reference if any.
     * @tparam T The type to be unreferenced.
     * @since 0.1.1
     */
    template <typename T>
    using Unref = typename std::remove_reference<T>::type;

    /**
     * Purifies a type from any reference, constness, volatile-ness or the like.
     * @tparam T The type to be purified.
     * @since 0.1.1
     */
    template <typename T>
    using Pure = typename std::remove_cv<typename std::remove_extent<Unref<T>>::type>::type;

    /**
     * Creates a type for a function so it is easier to pass them as arguments.
     * @tparam R The function's return value.
     * @tparam A The function's parameters types.
     * @since 0.1.1
     */
    template <typename R = void, typename ...A>
    using Function = R (*)(A...);
};

#endif