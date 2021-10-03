/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Common functional and logical operators implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>
#include <museqa/utility.hpp>

namespace museqa::utility
{
    /**
     * The logical AND operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The logical AND result between operands.
     */
    static inline constexpr const auto andl =
        [](auto x, auto y) -> bool { return bool(x) && bool(y); };

    /**
     * The logical OR operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The logical OR result between operands.
     */
    static inline constexpr const auto orl =
        [](auto x, auto y) -> bool { return bool(x) || bool(y); };

    /**
     * The less-than operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The less-than result between operands.
     */
    static inline constexpr const auto lt =
        [](auto x, auto y) -> bool { return x < y; };

    /**
     * The less-than-or-equal operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The less-than-or-equal result between operands.
     */
    static inline constexpr const auto lte =
        [](auto x, auto y) -> bool { return x <= y; };

    /**
     * The greater-than operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The greater-than result between operands.
     */
    static inline constexpr const auto gt =
        [](auto x, auto y) -> bool { return x > y; };

    /**
     * The greater-than-or-equal operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The greater-than-or-equal result between operands.
     */
    static inline constexpr const auto gte =
        [](auto x, auto y) -> bool { return x >= y; };

    /**
     * The equality operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The equality result between operands.
     */
    static inline constexpr const auto equ =
        [](auto x, auto y) -> bool { return x == y; };

    /**
     * The addition operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The addition result between operands.
     */
    static inline constexpr const auto add =
        [](auto x, auto y) { return x + y; };

    /**
     * The reversed addition operator.
     * @param x The second operand value.
     * @param y The first operand value.
     * @return The addition result between operands.
     */
    static inline constexpr const auto radd =
        [](auto x, auto y) { return y + x; };

    /**
     * The subtraction operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The subtraction result between operands.
     */
    static inline constexpr const auto sub =
        [](auto x, auto y) { return x - y; };

    /**
     * The reversed subtraction operator.
     * @param x The second operand value.
     * @param y The first operand value.
     * @return The subtraction result between operands.
     */
    static inline constexpr const auto rsub =
        [](auto x, auto y) { return y - x; };

    /**
     * The multiplication operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The multiplication result between operands.
     */
    static inline constexpr const auto mul =
        [](auto x, auto y) { return x * y; };

    /**
     * The reversed multiplication operator.
     * @param x The second operand value.
     * @param y The first operand value.
     * @return The multiplication result between operands.
     */
    static inline constexpr const auto rmul =
        [](auto x, auto y) { return y * x; };

    /**
     * The division operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The division result between operands.
     */
    static inline constexpr const auto div =
        [](auto x, auto y) { return x / y; };

    /**
     * The reversed division operator.
     * @param x The second operand value.
     * @param y The first operand value.
     * @return The division result between operands.
     */
    static inline constexpr const auto rdiv =
        [](auto x, auto y) { return y / x; };

    /**
     * The modulo operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The modulo result between operands.
     */
    static inline constexpr const auto mod =
        [](auto x, auto y) { return x % y; };

    /**
     * The reversed modulo operator.
     * @param x The sedond operand value.
     * @param y The first operand value.
     * @return The modulo result between operands.
     */
    static inline constexpr const auto rmod =
        [](auto x, auto y) { return y % x; };

    /**
     * The minimum operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @return The minimum between the operands.
     */
    static inline constexpr const auto min =
        [](auto x, auto y) { return x <= y ? x : y; };

    /**
     * The maximum operator.
     * @param x The first operand value.
     * @param y The second operand value.
     * @param z The list of other operand values.
     * @return The maximum between the operands.
     */
    static inline constexpr const auto max =
        [](auto x, auto y) { return x >= y ? x : y; };

    /**
     * The all-truthy validation operator.
     * @param x The list of parameters to be checked.
     * @return Are all given values truthy?
     */
    static inline constexpr const auto all =
        [](auto ...x) -> bool { return (bool(x) && ... && true); };

    /**
     * The any-truthy validation operator.
     * @param x The list of parameters to be checked.
     * @return Is there at least one given value that is truth-y?
     */
    static inline constexpr const auto any =
        [](auto ...x) -> bool { return (bool(x) || ... || false); };

    /**
     * The no-truthy validation operator.
     * @param x The list of parameters to be checked.
     * @return Are all given values false-y?
     */
    static inline constexpr const auto none =
        [](auto ...x) -> bool { return !any(x...); };
}
