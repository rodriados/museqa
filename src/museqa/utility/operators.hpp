/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Common functional and logical operators implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>
#include <museqa/utility.hpp>

MUSEQA_DISABLE_NVCC_WARNING_BEGIN(1835)
MUSEQA_DISABLE_NVCC_WARNING_BEGIN(20011)
MUSEQA_DISABLE_NVCC_WARNING_BEGIN(20012)
MUSEQA_DISABLE_GCC_WARNING_BEGIN("-Wattributes")

MUSEQA_BEGIN_NAMESPACE

namespace utility
{
    /**
     * The logical AND operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The logical AND result between operands.
     */
    __host__ __device__ inline static constexpr struct {
        template <typename X, typename Y>
        __host__ __device__ inline constexpr auto operator()
            (const X& x, const Y& y) const -> bool { return bool(x) && bool(y); }
    } andl;

    /**
     * The logical OR operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The logical OR result between operands.
     */
    __host__ __device__ inline static constexpr struct {
        template <typename X, typename Y>
        __host__ __device__ inline constexpr auto operator()
            (const X& x, const Y& y) const -> bool { return bool(x) || bool(y); }
    } orl;

    /**
     * The less-than operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The less-than result between operands.
     */
    __host__ __device__ inline static constexpr struct {
        template <typename X, typename Y>
        __host__ __device__ inline constexpr auto operator()
            (const X& x, const Y& y) const -> bool { return x < y; }
    } lt;

    /**
     * The less-than-or-equal operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The less-than-or-equal result between operands.
     */
    __host__ __device__ inline static constexpr struct {
        template <typename X, typename Y>
        __host__ __device__ inline constexpr auto operator()
            (const X& x, const Y& y) const -> bool { return x <= y; }
    } lte;

    /**
     * The greater-than operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The greater-than result between operands.
     */
    __host__ __device__ inline static constexpr struct {
        template <typename X, typename Y>
        __host__ __device__ inline constexpr auto operator()
            (const X& x, const Y& y) const -> bool { return x > y; }
    } gt;

    /**
     * The greater-than-or-equal operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The greater-than-or-equal result between operands.
     */
    __host__ __device__ inline static constexpr struct {
        template <typename X, typename Y>
        __host__ __device__ inline constexpr auto operator()
            (const X& x, const Y& y) const -> bool { return x >= y; }
    } gte;

    /**
     * The equality operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The equality result between operands.
     */
    __host__ __device__ inline static constexpr struct {
        template <typename X, typename Y>
        __host__ __device__ inline constexpr auto operator()
            (const X& x, const Y& y) const -> bool { return x == y; }
    } equ;

    /**
     * The addition operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The addition result between operands.
     */
    __host__ __device__ inline static constexpr struct {
        template <typename X, typename Y>
        __host__ __device__ inline constexpr auto operator()
            (const X& x, const Y& y) const -> decltype(auto) { return x + y; }
    } add;

    /**
     * The reversed addition operator.
     * @param x The second operand value.
     * @param y The first operand value.
     * @return The addition result between operands.
     */
    __host__ __device__ inline static constexpr struct {
        template <typename X, typename Y>
        __host__ __device__ inline constexpr auto operator()
            (const X& x, const Y& y) const -> decltype(auto) { return y + x; }
    } radd;

    /**
     * The subtraction operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The subtraction result between operands.
     */
    __host__ __device__ inline static constexpr struct {
        template <typename X, typename Y>
        __host__ __device__ inline constexpr auto operator()
            (const X& x, const Y& y) const -> decltype(auto) { return x - y; }
    } sub;

    /**
     * The reversed subtraction operator.
     * @param x The second operand value.
     * @param y The first operand value.
     * @return The subtraction result between operands.
     */
    __host__ __device__ inline static constexpr struct {
        template <typename X, typename Y>
        __host__ __device__ inline constexpr auto operator()
            (const X& x, const Y& y) const -> decltype(auto) { return y - x; }
    } rsub;

    /**
     * The multiplication operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The multiplication result between operands.
     */
    __host__ __device__ inline static constexpr struct {
        template <typename X, typename Y>
        __host__ __device__ inline constexpr auto operator()
            (const X& x, const Y& y) const -> decltype(auto) { return x * y; }
    } mul;

    /**
     * The reversed multiplication operator.
     * @param x The second operand value.
     * @param y The first operand value.
     * @return The multiplication result between operands.
     */
    __host__ __device__ inline static constexpr struct {
        template <typename X, typename Y>
        __host__ __device__ inline constexpr auto operator()
            (const X& x, const Y& y) const -> decltype(auto) { return y * x; }
    } rmul;

    /**
     * The division operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The division result between operands.
     */
    __host__ __device__ inline static constexpr struct {
        template <typename X, typename Y>
        __host__ __device__ inline constexpr auto operator()
            (const X& x, const Y& y) const -> decltype(auto) { return x / y; }
    } div;

    /**
     * The reversed division operator.
     * @param x The second operand value.
     * @param y The first operand value.
     * @return The division result between operands.
     */
    __host__ __device__ inline static constexpr struct {
        template <typename X, typename Y>
        __host__ __device__ inline constexpr auto operator()
            (const X& x, const Y& y) const -> decltype(auto) { return y / x; }
    } rdiv;

    /**
     * The modulo operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The modulo result between operands.
     */
    __host__ __device__ inline static constexpr struct {
        template <typename X, typename Y>
        __host__ __device__ inline constexpr auto operator()
            (const X& x, const Y& y) const -> decltype(auto) { return x % y; }
    } mod;

    /**
     * The reversed modulo operator.
     * @param x The sedond operand value.
     * @param y The first operand value.
     * @return The modulo result between operands.
     */
    __host__ __device__ inline static constexpr struct {
        template <typename X, typename Y>
        __host__ __device__ inline constexpr auto operator()
            (const X& x, const Y& y) const -> decltype(auto) { return y % x; }
    } rmod;

    /**
     * The minimum operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The minimum between the operands.
     */
    __host__ __device__ inline static constexpr struct {
        template <typename X, typename Y>
        __host__ __device__ inline constexpr auto operator()
            (const X& x, const Y& y) const -> decltype(auto) { return x <= y ? x : y; }
    } min;

    /**
     * The maximum operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @param z The list of other operand values.
     * @return The maximum between the operands.
     */
    __host__ __device__ inline static constexpr struct {
        template <typename X, typename Y>
        __host__ __device__ inline constexpr auto operator()
            (const X& x, const Y& y) const -> decltype(auto) { return x >= y ? x : y; }
    } max;

    /**
     * The all-truthy validation operator.
     * @param x The list of parameters to be checked.
     * @return Are all given values truthy?
     */
    __host__ __device__ inline static constexpr struct {
        template <typename ...X>
        __host__ __device__ inline constexpr auto operator()
            (const X&... x) const -> bool { return (bool(x) && ... && true); }
    } all;

    /**
     * The any-truthy validation operator.
     * @param x The list of parameters to be checked.
     * @return Is there at least one given value that is truth-y?
     */
    __host__ __device__ inline static constexpr struct {
        template <typename ...X>
        __host__ __device__ inline constexpr auto operator()
            (const X&... x) const -> bool { return (bool(x) || ... || false); }
    } any;

    /**
     * The no-truthy validation operator.
     * @param x The list of parameters to be checked.
     * @return Are all given values false-y?
     */
    __host__ __device__ inline static constexpr struct {
        template <typename ...X>
        __host__ __device__ inline constexpr auto operator()
            (const X&... x) const -> bool { return !any(x...); }
    } none;
}

MUSEQA_END_NAMESPACE

MUSEQA_DISABLE_GCC_WARNING_END("-Wattributes")
MUSEQA_DISABLE_NVCC_WARNING_END(20012)
MUSEQA_DISABLE_NVCC_WARNING_END(20011)
MUSEQA_DISABLE_NVCC_WARNING_END(1835)
