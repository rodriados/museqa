/** 
 * Multiple Sequence Alignment operators header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef OPERATOR_HPP_INCLUDED
#define OPERATOR_HPP_INCLUDED

#include "utils.hpp"

namespace op
{
    /**
     * Wraps an operator function. An operator always transforms two elements
     * of the same type into a single new value.
     * @tparam T The type upon which the operator works.
     * @since 0.1.1
     */
    template <typename T>
    struct Operator : public Functor<T(const T&, const T&)>
    {
        using Functor<T(const T&, const T&)>::Functor;
        using Functor<T(const T&, const T&)>::operator=;
    };

    /**
     * The logical AND operator.
     * @return The logical AND result between operands.
     */
    __host__ __device__ inline constexpr auto andl(bool a, bool b) noexcept -> bool
    {
        return a && b;
    }

    /**
     * The logical OR operator.
     * @return The logical OR result between operands.
     */
    __host__ __device__ inline constexpr auto orl(bool a, bool b) noexcept -> bool
    {
        return a || b;
    }

    /**
     * The logical less-than operator.
     * @return The logical result between operands.
     */
    template <typename T>
    __host__ __device__ inline constexpr auto lt(const T& a, const T& b) noexcept -> bool
    {
        return a < b;
    }

    /**
     * The logical less-than-or-equal operator.
     * @return The logical result between operands.
     */
    template <typename T>
    __host__ __device__ inline constexpr auto lte(const T& a, const T& b) noexcept -> bool
    {
        return a <= b;
    }

    /**
     * The logical greater-than operator.
     * @return The logical result between operands.
     */
    template <typename T>
    __host__ __device__ inline constexpr auto gt(const T& a, const T& b) noexcept -> bool
    {
        return a > b;
    }

    /**
     * The logical greater-than-or-equal operator.
     * @return The logical result between operands.
     */
    template <typename T>
    __host__ __device__ inline constexpr auto gte(const T& a, const T& b) noexcept -> bool
    {
        return a >= b;
    }

    /**
     * The logical equal operator.
     * @return The logical result between operands.
     */
    template <typename T>
    __host__ __device__ inline constexpr auto eq(const T& a, const T& b) noexcept -> bool
    {
        return a == b;
    }

    /**
     * The sum operator.
     * @return The sum of the operands.
     */
    template <typename T>
    __host__ __device__ inline constexpr auto add(const T& a, const T& b) noexcept -> decltype(a + b)
    {
        return a + b;
    }

    /**
     * The subtraction operator.
     * @return The difference of the operands.
     */
    template <typename T>
    __host__ __device__ inline constexpr auto sub(const T& a, const T& b) noexcept -> decltype(a - b)
    {
        return a - b;
    }

    /**
     * The multiplication operator.
     * @return The product of the operands.
     */
    template <typename T>
    __host__ __device__ inline constexpr auto mul(const T& a, const T& b) noexcept -> decltype(a * b)
    {
        return a * b;
    }

    /**
     * The division operator.
     * @return The division of the operands.
     */
    template <typename T>
    __host__ __device__ inline constexpr auto div(const T& a, const T& b) noexcept -> decltype(a / b)
    {
        return a / b;
    }

    /**
     * The module operator.
     * @return The module of the operands.
     */
    template <typename T>
    __host__ __device__ inline constexpr auto mod(const T& a, const T& b) noexcept -> decltype(a % b)
    {
        return a % b;
    }

    /**
     * The minimum operator.
     * @return The minimum between the operands.
     */
    template <typename T>
    __host__ __device__ inline constexpr auto min(const T& a, const T& b) noexcept -> decltype(a < b ? a : b)
    {
        return a < b ? a : b;
    }

    /**
     * The maximum operator.
     * @return The maximum between the operands.
     */
    template <typename T>
    __host__ __device__ inline constexpr auto max(const T& a, const T& b) noexcept -> decltype(a > b ? a : b)
    {
        return a > b ? a : b;
    }
};

#endif