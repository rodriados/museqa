/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Common functional and logical operators implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>
#include <museqa/utility.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace utility
{
    /**
     * The logical-and operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The logical-and result between operands.
     */
    __device__ MUSEQA_CONSTEXPR static struct {
        template <typename X, typename Y> MUSEQA_CUDA_CONSTEXPR
        bool operator()(X&& x, Y&& y) const noexcept { return x && y; }
    } andL;

    /**
     * The logical-or operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The logical-or result between operands.
     */
    __device__ MUSEQA_CONSTEXPR static struct {
        template <typename X, typename Y> MUSEQA_CUDA_CONSTEXPR
        bool operator()(X&& x, Y&& y) const noexcept { return x || y; }
    } orL;

    /**
     * The less-than operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The less-than result between operands.
     */
    __device__ MUSEQA_CONSTEXPR static struct {
        template <typename X, typename Y> MUSEQA_CUDA_CONSTEXPR
        bool operator()(X&& x, Y&& y) const noexcept { return x < y; }
    } lt, less;

    /**
     * The less-than-or-equal operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The less-than-or-equal result between operands.
     */
    __device__ MUSEQA_CONSTEXPR static struct {
        template <typename X, typename Y> MUSEQA_CUDA_CONSTEXPR
        bool operator()(X&& x, Y&& y) const noexcept { return x <= y; }
    } lte, less_equal;

    /**
     * The greater-than operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The greater-than result between operands.
     */
    __device__ MUSEQA_CONSTEXPR static struct {
        template <typename X, typename Y> MUSEQA_CUDA_CONSTEXPR
        bool operator()(X&& x, Y&& y) const noexcept { return x > y; }
    } gt, greater;

    /**
     * The greater-than-or-equal operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The greater-than-or-equal result between operands.
     */
    __device__ MUSEQA_CONSTEXPR static struct {
        template <typename X, typename Y> MUSEQA_CUDA_CONSTEXPR
        bool operator()(X&& x, Y&& y) const noexcept { return x >= y; }
    } gte, greater_equal;

    /**
     * The equality operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The equality result between operands.
     */
    __device__ MUSEQA_CONSTEXPR static struct {
        template <typename X, typename Y> MUSEQA_CUDA_CONSTEXPR
        bool operator()(X&& x, Y&& y) const noexcept { return x == y; }
    } equ, equal;

    /**
     * The addition operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The addition result between operands.
     */
    __device__ MUSEQA_CONSTEXPR static struct {
        template <typename X, typename Y> MUSEQA_CUDA_CONSTEXPR
        auto operator()(X&& x, Y&& y) const noexcept { return x + y; }
    } add;

    /**
     * The reversed addition operator.
     * @param x The second operand value.
     * @param y The first operand value.
     * @return The addition result between operands.
     */
    __device__ MUSEQA_CONSTEXPR static struct {
        template <typename X, typename Y> MUSEQA_CUDA_CONSTEXPR
        auto operator()(X&& x, Y&& y) const noexcept { return y + x; }
    } radd;

    /**
     * The subtraction operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The subtraction result between operands.
     */
    __device__ MUSEQA_CONSTEXPR static struct {
        template <typename X, typename Y> MUSEQA_CUDA_CONSTEXPR
        auto operator()(X&& x, Y&& y) const noexcept { return x - y; }
    } sub;

    /**
     * The reversed subtraction operator.
     * @param x The second operand value.
     * @param y The first operand value.
     * @return The subtraction result between operands.
     */
    __device__ MUSEQA_CONSTEXPR static struct {
        template <typename X, typename Y> MUSEQA_CUDA_CONSTEXPR
        auto operator()(X&& x, Y&& y) const noexcept { return y - x; }
    } rsub;

    /**
     * The multiplication operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The multiplication result between operands.
     */
    __device__ MUSEQA_CONSTEXPR static struct {
        template <typename X, typename Y> MUSEQA_CUDA_CONSTEXPR
        auto operator()(X&& x, Y&& y) const noexcept { return x * y; }
    } mul;

    /**
     * The reversed multiplication operator.
     * @param x The second operand value.
     * @param y The first operand value.
     * @return The multiplication result between operands.
     */
    __device__ MUSEQA_CONSTEXPR static struct {
        template <typename X, typename Y> MUSEQA_CUDA_CONSTEXPR
        auto operator()(X&& x, Y&& y) const noexcept { return y * x; }
    } rmul;

    /**
     * The division operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The division result between operands.
     */
    __device__ MUSEQA_CONSTEXPR static struct {
        template <typename X, typename Y> MUSEQA_CUDA_CONSTEXPR
        auto operator()(X&& x, Y&& y) const noexcept { return x / y; }
    } div;

    /**
     * The reversed division operator.
     * @param x The second operand value.
     * @param y The first operand value.
     * @return The division result between operands.
     */
    __device__ MUSEQA_CONSTEXPR static struct {
        template <typename X, typename Y> MUSEQA_CUDA_CONSTEXPR
        auto operator()(X&& x, Y&& y) const noexcept { return y / x; }
    } rdiv;

    /**
     * The modulo operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The modulo result between operands.
     */
    __device__ MUSEQA_CONSTEXPR static struct {
        template <typename X, typename Y> MUSEQA_CUDA_CONSTEXPR
        auto operator()(X&& x, Y&& y) const noexcept { return x % y; }
    } mod;

    /**
     * The reversed modulo operator.
     * @param x The sedond operand value.
     * @param y The first operand value.
     * @return The modulo result between operands.
     */
    __device__ MUSEQA_CONSTEXPR static struct {
        template <typename X, typename Y> MUSEQA_CUDA_CONSTEXPR
        auto operator()(X&& x, Y&& y) const noexcept { return y % x; }
    } rmod;

    /**
     * The minimum operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The minimum between the operands.
     */
    __device__ MUSEQA_CONSTEXPR static struct {
        template <typename X, typename Y> MUSEQA_CUDA_CONSTEXPR
        auto operator()(X&& x, Y&& y) const noexcept { return x <= y ? x : y; }
    } min;

    /**
     * The maximum operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @param z The list of other operand values.
     * @return The maximum between the operands.
     */
    __device__ MUSEQA_CONSTEXPR static struct {
        template <typename X, typename Y> MUSEQA_CUDA_CONSTEXPR
        auto operator()(X&& x, Y&& y) const noexcept { return x >= y ? x : y; }
    } max;

    /**
     * The all-truthy validation operator.
     * @param x The list of parameters to be checked.
     * @return Are all given values truthy?
     */
    __device__ MUSEQA_CONSTEXPR static struct {
        template <typename ...X> MUSEQA_CUDA_CONSTEXPR
        bool operator()(X&&... x) const noexcept { return (x && ... && true); }
    } all;

    /**
     * The any-truthy validation operator.
     * @param x The list of parameters to be checked.
     * @return Is there at least one given value that is truth-y?
     */
    __device__ MUSEQA_CONSTEXPR static struct {
        template <typename ...X> MUSEQA_CUDA_CONSTEXPR
        bool operator()(X&&... x) const noexcept { return (x || ... || false); }
    } any;

    /**
     * The no-truthy validation operator.
     * @param x The list of parameters to be checked.
     * @return Are all given values false-y?
     */
    __device__ MUSEQA_CONSTEXPR static struct {
        template <typename ...X> MUSEQA_CUDA_CONSTEXPR
        bool operator()(X&&... x) const noexcept { return !any(x...); }
    } none;
}

MUSEQA_END_NAMESPACE
