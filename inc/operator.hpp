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
    __host__ __device__ inline constexpr auto andl(const bool& a, const bool& b) noexcept
    -> bool { return a && b; }

    /**
     * The logical OR operator.
     * @return The logical OR result between operands.
     */
    __host__ __device__ inline constexpr auto orl(const bool& a, const bool& b) noexcept
    -> bool { return a || b; }

    /**
     * The sum operator.
     * @return The sum of the operands.
     */
    template <typename T>
    __host__ __device__ inline constexpr auto add(const T& a, const T& b) noexcept
    -> T { return a + b; }

    /**
     * The subtraction operator.
     * @return The difference of the operands.
     */
    template <typename T>
    __host__ __device__ inline constexpr auto sub(const T& a, const T& b) noexcept
    -> T { return a - b; }

    /**
     * The multiplication operator.
     * @return The product of the operands.
     */
    template <typename T>
    __host__ __device__ inline constexpr auto mul(const T& a, const T& b) noexcept
    -> T { return a * b; }

    /**
     * The division operator.
     * @return The division of the operands.
     */
    template <typename T>
    __host__ __device__ inline constexpr auto div(const T& a, const T& b) noexcept
    -> T { return a / b; }

    /**
     * The module operator.
     * @return The module of the operands.
     */
    template <typename T>
    __host__ __device__ inline constexpr auto mod(const T& a, const T& b) noexcept
    -> T { return a % b; }

    /**
     * The minimum operator.
     * @return The minimum between the operands.
     */
    template <typename T>
    __host__ __device__ inline constexpr auto min(const T& a, const T& b) noexcept
    -> T { return a < b ? a : b; }

    /**
     * The maximum operator.
     * @return The maximum between the operands.
     */
    template <typename T>
    __host__ __device__ inline constexpr auto max(const T& a, const T& b) noexcept
    -> T { return a > b ? a : b; }

    /**
     * The logical AND operator functor.
     * @since 0.1.1
     */
    struct And : public Operator<bool>
    {   __host__ __device__ inline constexpr And() noexcept
        :   Operator<bool> {andl} {}
    };

    /**
     * The logical OR operator.
     * @since 0.1.1
     */
    struct Or : public Operator<bool>
    {   __host__ __device__ inline constexpr Or() noexcept
        :   Operator<bool> {orl} {}
    };

    /**
     * The sum operator functor.
     * @since 0.1.1
     */
    template <typename T>
    struct Add : public Operator<T>
    {   __host__ __device__ inline constexpr Add() noexcept
        :   Operator<T> {add} {}
    };

    /**
     * The subtraction operator functor.
     * @since 0.1.1
     */
    template <typename T>
    struct Sub : public Operator<T>
    {   __host__ __device__ inline constexpr Sub() noexcept
        :   Operator<T> {sub} {}
    };

    /**
     * The multiplication operator functor.
     * @since 0.1.1
     */
    template <typename T>
    struct Mul : public Operator<T>
    {   __host__ __device__ inline constexpr Mul() noexcept
        :   Operator<T> {mul} {}
    };

    /**
     * The division operator functor.
     * @since 0.1.1
     */
    template <typename T>
    struct Div : public Operator<T>
    {   __host__ __device__ inline constexpr Div() noexcept
        :   Operator<T> {div} {}
    };

    /**
     * The module operator functor.
     * @since 0.1.1
     */
    template <typename T>
    struct Mod : public Operator<T>
    {   __host__ __device__ inline constexpr Mod() noexcept
        :   Operator<T> {mod} {}
    };

    /**
     * The minimum operator functor.
     * @since 0.1.1
     */
    template <typename T>
    struct Min : public Operator<T>
    {   __host__ __device__ inline constexpr Min() noexcept
        :   Operator<T> {min} {}
    };

    /**
     * The maximum operator functor.
     * @since 0.1.1
     */
    template <typename T>
    struct Max : public Operator<T>
    {   __host__ __device__ inline constexpr Max() noexcept
        :   Operator<T> {max} {}
    };
};

#endif