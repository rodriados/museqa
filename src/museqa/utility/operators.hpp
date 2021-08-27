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
         * A macro for generating the anonymous type of an operator on its definition.
         * @param f The operation to be wrapped by an operator type.
         * @since 1.0
         */
        #define museqa_binary_operator_type(f) struct {                                             \
            template <typename X, typename Y>                                                       \
            __host__ __device__ inline constexpr auto operator()(const X& x, const Y& y) const      \
            noexcept -> decltype((f)) {                                                             \
                return (f);                                                                         \
            }                                                                                       \
        }

        namespace
        {
            /**
             * The logical AND operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @return The logical AND result between operands.
             */
            static constexpr const museqa_binary_operator_type((x && y)) andl, logic_and;

            /**
             * The logical OR operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @return The logical OR result between operands.
             */
            static constexpr const museqa_binary_operator_type((x || y)) orl, logic_or;

            /**
             * The less-than operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @return The less-than result between operands.
             */
            static constexpr const museqa_binary_operator_type((x < y)) lt, less;

            /**
             * The greater-than operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @return The greater-than result between operands.
             */
            static constexpr const museqa_binary_operator_type((x > y)) gt, greater;

            /**
             * The less-than-or-equal operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @return The less-than-or-equal result between operands.
             */
            static constexpr const museqa_binary_operator_type((x <= y)) lte, less_equal;

            /**
             * The greater-than-or-equal operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @return The greater-than-or-equal result between operands.
             */
            static constexpr const museqa_binary_operator_type((x >= y)) gte, greater_equal;

            /**
             * The equality operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @return The equality result between operands.
             */
            static constexpr const museqa_binary_operator_type((x == y)) equ, equals;

            /**
             * The addition operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @return The addition result between operands.
             */
            static constexpr const museqa_binary_operator_type((x + y)) add;

            /**
             * The subtraction operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @return The subtraction result between operands.
             */
            static constexpr const museqa_binary_operator_type((x - y)) sub;

            /**
             * The multiplication operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @return The multiplication result between operands.
             */
            static constexpr const museqa_binary_operator_type((x * y)) mul;

            /**
             * The division operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @return The division result between operands.
             */
            static constexpr const museqa_binary_operator_type((x / y)) div;

            /**
             * The modulo operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @return The modulo result between operands.
             */
            static constexpr const museqa_binary_operator_type((x % y)) mod;

            /**
             * The minimum operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @param z The list of other operand values.
             * @return The minimum between the operands.
             */
            static constexpr const struct {
                template <typename T>
                __host__ __device__ inline constexpr auto operator()(const T& x) const
                noexcept -> const T& { return x; }

                template <typename T, typename ...U>
                __host__ __device__ inline constexpr auto operator()(const T& x, const T& y, const U&... z) const
                noexcept -> const T& { return operator()(x <= y ? x : y, z...); }
            } min;

            /**
             * The maximum operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @param z The list of other operand values.
             * @return The maximum between the operands.
             */
            static constexpr const struct {
                template <typename T>
                __host__ __device__ inline constexpr auto operator()(const T& x) const
                noexcept -> const T& { return x; }

                template <typename T, typename ...U>
                __host__ __device__ inline constexpr auto operator()(const T& x, const T& y, const U&... z) const
                noexcept -> const T& { return operator()(x >= y ? x : y, z...); }
            } max;

            /**
             * The all-truthy validation operator.
             * @param x The first parameter in line to be checked.
             * @param z The list of following parameters to be checked.
             * @return Are all given values truthy?
             */
            static constexpr const struct {
                __host__ __device__ inline constexpr auto operator()() const
                noexcept -> bool { return true; }

                template <typename ...T>
                __host__ __device__ inline constexpr auto operator()(bool x, T&&... z) const
                noexcept -> bool { return x && operator()(z...); }
            } all;

            /**
             * The any-truthy validation operator.
             * @param x The first parameter in line to be checked.
             * @param z The list of following parameters to be checked.
             * @return Is there at least one given value that is truth-y?
             */
            static constexpr const struct {
                __host__ __device__ inline constexpr auto operator()() const
                noexcept -> bool { return false; }

                template <typename ...T>
                __host__ __device__ inline constexpr auto operator()(bool x, T&&... z) const
                noexcept -> bool { return x || operator()(z...); }
            } any;

            /**
             * The no-truthy validation operator.
             * @param z The list of parameters to be checked.
             * @return Are all given values false-y?
             */
            static constexpr const struct {
                template <typename ...T>
                __host__ __device__ inline constexpr auto operator()(T&&... z) const
                noexcept -> bool { return !any(z...); }
            } none;
        }

        #undef museqa_binary_operator_type
    }
}
