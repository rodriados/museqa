/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Common functional and logical operators implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>
#include <museqa/utility.hpp>

namespace museqa
{
    namespace utility
    {
        namespace
        {
            /*
             * A macro for defining operators both in host and device at the same
             * time. As we cannot create constexpr variable on device, we place
             * our operators in constant memory.
             */
          #if !defined(MUSEQA_RUNTIME_DEVICE)
            #define __immutable__ constexpr
          #else
            #define __immutable__ __constant__
          #endif

            /**
             * A macro for generating the operation method of an operator on its definition.
             * @param f The operation to be wrapped by an operator type.
             * @since 1.0
             */
            #define binary_operation(f)                                                             \
                template <typename X, typename Y>                                                   \
                __host__ __device__ inline constexpr auto operator()(const X& x, const Y& y) const  \
                noexcept -> decltype((f)) {                                                         \
                    return (f);                                                                     \
                }

            /**
             * The logical AND operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @return The logical AND result between operands.
             */
            __immutable__ static const struct { binary_operation(x && y) } andl;

            /**
             * The logical OR operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @return The logical OR result between operands.
             */
            __immutable__ static const struct { binary_operation(x || y) } orl;

            /**
             * The less-than operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @return The less-than result between operands.
             */
            __immutable__ static const struct { binary_operation(x < y) } lt, less;

            /**
             * The greater-than operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @return The greater-than result between operands.
             */
            __immutable__ static const struct { binary_operation(x > y) } gt, greater;

            /**
             * The less-than-or-equal operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @return The less-than-or-equal result between operands.
             */
            __immutable__ static const struct { binary_operation(x <= y) } lte;

            /**
             * The greater-than-or-equal operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @return The greater-than-or-equal result between operands.
             */
            __immutable__ static const struct { binary_operation(x >= y) } gte;

            /**
             * The equality operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @return The equality result between operands.
             */
            __immutable__ static const struct { binary_operation(x == y) } equ, equals;

            /**
             * The addition operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @return The addition result between operands.
             */
            __immutable__ static const struct { binary_operation(x + y) } add;

            /**
             * The reversed addition operator.
             * @param x The second operand value.
             * @param y The first operand value.
             * @return The addition result between operands.
             */
            __immutable__ static const struct { binary_operation(y + x) } radd;

            /**
             * The subtraction operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @return The subtraction result between operands.
             */
            __immutable__ static const struct { binary_operation(x - y) } sub;

            /**
             * The reversed subtraction operator.
             * @param x The second operand value.
             * @param y The first operand value.
             * @return The subtraction result between operands.
             */
            __immutable__ static const struct { binary_operation(y - x) } rsub;

            /**
             * The multiplication operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @return The multiplication result between operands.
             */
            __immutable__ static const struct { binary_operation(x * y) } mul;

            /**
             * The reversed multiplication operator.
             * @param x The second operand value.
             * @param y The first operand value.
             * @return The multiplication result between operands.
             */
            __immutable__ static const struct { binary_operation(y * x) } rmul;

            /**
             * The division operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @return The division result between operands.
             */
            __immutable__ static const struct { binary_operation(x / y) } div;

            /**
             * The reversed division operator.
             * @param x The second operand value.
             * @param y The first operand value.
             * @return The division result between operands.
             */
            __immutable__ static const struct { binary_operation(y / x) } rdiv;

            /**
             * The modulo operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @return The modulo result between operands.
             */
            __immutable__ static const struct { binary_operation(x % y) } mod;

            /**
             * The reversed modulo operator.
             * @param x The sedond operand value.
             * @param y The first operand value.
             * @return The modulo result between operands.
             */
            __immutable__ static const struct { binary_operation(y % x) } rmod;

            /**
             * The minimum operator.
             * @param x The first operand value.
             * @param y The second operand value.
             * @param z The list of other operand values.
             * @return The minimum between the operands.
             */
            __immutable__ static const struct {
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
            __immutable__ static const struct {
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
            __immutable__ static const struct {
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
            __immutable__ static const struct {
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
            __immutable__ static const struct {
                template <typename ...T>
                __host__ __device__ inline constexpr auto operator()(T&&... z) const
                noexcept -> bool { return !any(z...); }
            } none;

            #undef binary_operation
            #undef __immutable__
        }
    }
}
