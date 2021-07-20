/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Common functional and logical operators implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/utility.hpp>

namespace museqa
{
    namespace utility
    {
        /**
         * The logical AND operator.
         * @tparam X The first operand type.
         * @tparam Y The second operand type.
         * @param x The first operand value.
         * @param y The second operand value.
         * @return The logical AND result between operands.
         */
        template <typename X, typename Y = X>
        __host__ __device__ inline static constexpr auto andl(const X& x, const Y& y) noexcept
        -> bool { return x && y; }

        /**
         * The logical OR operator.
         * @tparam X The first operand type.
         * @tparam Y The second operand type.
         * @param x The first operand value.
         * @param y The second operand value.
         * @return The logical OR result between operands.
         */
        template <typename X, typename Y = X>
        __host__ __device__ inline static constexpr auto orl(const X& x, const Y& y) noexcept
        -> bool { return x || y; }

        /**
         * The less-than operator.
         * @tparam X The first operand type.
         * @tparam Y The second operand type.
         * @param x The first operand value.
         * @param y The second operand value.
         * @return The less-than result between operands.
         */
        template <typename X, typename Y = X>
        __host__ __device__ inline static constexpr auto lt(const X& x, const Y& y) noexcept
        -> bool { return x < y; }

        /**
         * The greater-than operator.
         * @tparam X The first operand type.
         * @tparam Y The second operand type.
         * @param x The first operand value.
         * @param y The second operand value.
         * @return The greater-than result between operands.
         */
        template <typename X, typename Y = X>
        __host__ __device__ inline static constexpr auto gt(const X& x, const Y& y) noexcept
        -> bool { return x > y; }

        /**
         * The less-than-or-equal operator.
         * @tparam X The first operand type.
         * @tparam Y The second operand type.
         * @param x The first operand value.
         * @param y The second operand value.
         * @return The less-than-or-equal result between operands.
         */
        template <typename X, typename Y = X>
        __host__ __device__ inline static constexpr auto lte(const X& x, const Y& y) noexcept
        -> bool { return x <= y; }

        /**
         * The greater-than-or-equal operator.
         * @tparam X The first operand type.
         * @tparam Y The second operand type.
         * @param x The first operand value.
         * @param y The second operand value.
         * @return The greater-than-or-equal result between operands.
         */
        template <typename X, typename Y = X>
        __host__ __device__ inline static constexpr auto gte(const X& x, const Y& y) noexcept
        -> bool { return x >= y; }

        /**
         * The equality operator.
         * @tparam X The first operand type.
         * @tparam Y The second operand type.
         * @param x The first operand value.
         * @param y The second operand value.
         * @return The equality result between operands.
         */
        template <typename X, typename Y = X>
        __host__ __device__ inline static constexpr auto eq(const X& x, const Y& y) noexcept
        -> bool { return x == y; }

        /**
         * The addition operator.
         * @tparam X The first operand type.
         * @tparam Y The second operand type.
         * @param x The first operand value.
         * @param y The second operand value.
         * @return The addition result between operands.
         */
        template <typename X, typename Y = X>
        __host__ __device__ inline static constexpr auto add(const X& x, const Y& y) noexcept
        -> decltype(x + y) { return x + y; }

        /**
         * The subtraction operator.
         * @tparam X The first operand type.
         * @tparam Y The second operand type.
         * @param x The first operand value.
         * @param y The second operand value.
         * @return The subtraction result between operands.
         */
        template <typename X, typename Y = X>
        __host__ __device__ inline static constexpr auto sub(const X& x, const Y& y) noexcept
        -> decltype(x - y) { return x - y; }

        /**
         * The multiplication operator.
         * @tparam X The first operand type.
         * @tparam Y The second operand type.
         * @param x The first operand value.
         * @param y The second operand value.
         * @return The multiplication result between operands.
         */
        template <typename X, typename Y = X>
        __host__ __device__ inline static constexpr auto mul(const X& x, const Y& y) noexcept
        -> decltype(x * y) { return x * y; }

        /**
         * The division operator.
         * @tparam X The first operand type.
         * @tparam Y The second operand type.
         * @param x The first operand value.
         * @param y The second operand value.
         * @return The division result between operands.
         */
        template <typename X, typename Y = X>
        __host__ __device__ inline static constexpr auto div(const X& x, const Y& y) noexcept
        -> decltype(x / y) { return x / y; }

        /**
         * The modulo operator.
         * @tparam X The first operand type.
         * @tparam Y The second operand type.
         * @param x The first operand value.
         * @param y The second operand value.
         * @return The modulo result between operands.
         */
        template <typename X, typename Y = X>
        __host__ __device__ inline static constexpr auto mod(const X& x, const Y& y) noexcept
        -> decltype(x % y) { return x % y; }

        /**
         * The minimum operator.
         * @tparam X The first operand type.
         * @tparam Y The second operand type.
         * @param x The first operand value.
         * @param y The second operand value.
         * @return The minimum between the operands.
         */
        template <typename X>
        __host__ __device__ inline static constexpr auto min(const X& x, const X& y) noexcept
        -> const X& { return x <= y ? x : y; }

        /**
         * The maximum operator.
         * @tparam X The first operand type.
         * @tparam Y The second operand type.
         * @param x The first operand value.
         * @param y The second operand value.
         * @return The maximum between the operands.
         */
        template <typename X>
        __host__ __device__ inline static constexpr auto max(const X& x, const X& y) noexcept
        -> const X& { return x >= y ? x : y; }

        /**#@+
         * Checks whether all given values are truth-y.
         * @tparam Y The given parameters' types.
         * @param x The first parameter in line to be checked.
         * @param y The list of following parameters to be checked.
         * @return Are all given values truth-y?
         */
        __host__ __device__ inline static constexpr auto all() noexcept
        -> bool { return true; }

        template <typename ...Y>
        __host__ __device__ inline static constexpr auto all(bool x, Y&&... y) noexcept
        -> bool { return x && all(y...); }
        /**#@-*/

        /**#@+
         * Checks whether at least one of the given values is truth-y.
         * @tparam Y The given parameters' types.
         * @param x The first parameter in line to be checked.
         * @param y The list of following parameters to be checked.
         * @return Is there at least one given value that is truth-y?
         */
        __host__ __device__ inline static constexpr auto any() noexcept
        -> bool { return false; }

        template <typename ...Y>
        __host__ __device__ inline static constexpr auto any(bool x, Y&&... y) noexcept
        -> bool { return x || any(y...); }
        /**#@-*/

        /**
         * Checks whether none of the given values is truth-y.
         * @tparam X The given parameters' types.
         * @param x The list of parameters to be checked.
         * @return Are all given values false-y?
         */
        template <typename ...X>
        __host__ __device__ inline static constexpr auto none(X&&... x) noexcept
        -> bool { return !any(x...); }
    }
}
