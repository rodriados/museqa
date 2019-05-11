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
     * The logical AND operator.
     * @return The logical AND result between operands.
     */
    static constexpr struct AndL {
        __host__ __device__ inline constexpr auto operator()(bool a, bool b) const noexcept
        -> bool { return a && b; }
    } const andl = {};

    /**
     * The logical OR operator.
     * @return The logical OR result between operands.
     */
    static constexpr struct OrL {
        __host__ __device__ inline constexpr auto operator()(bool a, bool b) const noexcept
        -> bool { return a || b; }
    } const orl = {};

    /**
     * The logical less-than operator.
     * @return The logical result between operands.
     */
    static constexpr struct LessT {
        template <typename T>
        __host__ __device__ inline constexpr auto operator()(const T& a, const T& b) const noexcept
        -> bool { return a < b; }
    } const lt = {};

    /**
     * The logical less-than-or-equal operator.
     * @return The logical result between operands.
     */
    static constexpr struct LessE {
        template <typename T>
        __host__ __device__ inline constexpr auto operator()(const T& a, const T& b) const noexcept
        -> bool { return a <= b; }
    } const lte = {};

    /**
     * The logical equal operator.
     * @return The logical result between operands.
     */
    static constexpr struct Equal {
        template <typename T>
        __host__ __device__ inline constexpr auto operator()(const T& a, const T& b) const noexcept
        -> bool { return a == b; }
    } const eq = {};

    /**
     * The sum operator.
     * @return The sum of the operands.
     */
    static constexpr struct Add {
        template <typename T>
        __host__ __device__ inline constexpr auto operator()(const T& a, const T& b) const noexcept
        -> decltype(a + b) { return a + b; }
    } const add = {};

    /**
     * The subtraction operator.
     * @return The difference of the operands.
     */
    static constexpr struct Sub {
        template <typename T>
        __host__ __device__ inline constexpr auto operator()(const T& a, const T& b) const noexcept
        -> decltype(a - b) { return a - b; }
    } const sub = {};

    /**
     * The multiplication operator.
     * @return The product of the operands.
     */
    static constexpr struct Mul {
        template <typename T>
        __host__ __device__ inline constexpr auto operator()(const T& a, const T& b) const noexcept
        -> decltype(a * b) { return a * b; }
    } const mul = {};

    /**
     * The division operator.
     * @return The division of the operands.
     */
    static constexpr struct Div {
        template <typename T>
        __host__ __device__ inline constexpr auto operator()(const T& a, const T& b) const noexcept
        -> decltype(a / b) { return a / b; }
    } const div = {};

    /**
     * The module operator.
     * @return The module of the operands.
     */
    static constexpr struct Mod {
        template <typename T>
        __host__ __device__ inline constexpr auto operator()(const T& a, const T& b) const noexcept
        -> decltype(a % b) { return a % b; }
    } const mod = {};

    /**
     * The minimum operator.
     * @return The minimum between the operands.
     */
    static constexpr struct Min {
        template <typename T>
        __host__ __device__ inline constexpr auto operator()(const T& a, const T& b) const noexcept
        -> decltype(a < b ? a : b) { return a < b ? a : b; }
    } const min = {};

    /**
     * The maximum operator.
     * @return The maximum between the operands.
     */
    static constexpr struct Max {
        template <typename T>
        __host__ __device__ inline constexpr auto operator()(const T& a, const T& b) const noexcept
        -> decltype(a > b ? a : b) { return a > b ? a : b; }
    } const max = {};
};

#endif