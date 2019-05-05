/** 
 * Multiple Sequence Alignment operators header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef OPERATOR_HPP_INCLUDED
#define OPERATOR_HPP_INCLUDED

namespace op
{
    /**
     * The logical AND operator.
     * @return The logical AND result between operands.
     */
    template <typename T, typename U>
    inline constexpr auto andl(const T& t, const U& u) noexcept
    -> decltype(t && u) { return t && u; }

    /**
     * The logical OR operator.
     * @return The logical OR result between operands.
     */
    template <typename T, typename U>
    inline constexpr auto orl(const T& t, const U& u) noexcept
    -> decltype(t || u) { return t || u; }

    /**
     * The addition operator.
     * @return The sum of the operands.
     */
    template <typename T, typename U>
    inline constexpr auto add(const T& t, const U& u) noexcept
    -> decltype(t + u) { return t + u; }

    /**
     * The subtraction operator.
     * @return The difference of the operands.
     */
    template <typename T, typename U>
    inline constexpr auto sub(const T& t, const U& u) noexcept
    -> decltype(t - u) { return t - u; }

    /**
     * The multiplication operator.
     * @return The product of the operands.
     */
    template <typename T, typename U>
    inline constexpr auto mul(const T& t, const U& u) noexcept
    -> decltype(t * u) { return t * u; }

    /**
     * The division operator.
     * @return The division of the operands.
     */
    template <typename T, typename U>
    inline constexpr auto div(const T& t, const U& u) noexcept
    -> decltype(t / u) { return t / u; }

    /**
     * The module operator.
     * @return The module of the operands.
     */
    template <typename T, typename U>
    inline constexpr auto mod(const T& t, const U& u) noexcept
    -> decltype(t % u) { return t % u; }

    /**
     * The minimum operator.
     * @return The minimum between the operands.
     */
    template <typename T>
    inline constexpr auto min(const T& a, const T& b) noexcept
    -> T { return a < b ? a : b; }

    /**
     * The maximum operator.
     * @return The maximum between the operands.
     */
    template <typename T>
    inline constexpr auto max(const T& a, const T& b) noexcept
    -> T { return a > b ? a : b; }
};

#endif