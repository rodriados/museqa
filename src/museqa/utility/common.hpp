/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Common utility functions and definitions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <utility>

#include <museqa/environment.h>

MUSEQA_BEGIN_NAMESPACE

namespace utility
{
    /**
     * Parses the given string value into any other generic type.
     * @tparam T The target type to parse the string to.
     * @param value The string to be parsed.
     * @return The parsed value to the requested type.
     */
    template <typename T>
    inline auto parse(const std::string& value)
    -> typename std::enable_if<std::is_convertible<std::string, T>::value, T>::type
    {
        return T (value);
    }

    /**
     * Parses the given string value into an integer type.
     * @tparam T The target type to parse the string to.
     * @param value The string to be parsed.
     * @return The parsed value to the requested type.
     * @throw std::exception Error detected during operation.
     */
    template <typename T>
    inline auto parse(const std::string& value)
    -> typename std::enable_if<std::is_integral<T>::value, T>::type
    {
        return static_cast<T>(std::stoull(value));
    }

    /**
     * Parses the given string value into a floating-point type.
     * @tparam T The target type to parse the string to.
     * @param value The string to be parsed.
     * @return The parsed value to the requested type.
     * @throw std::exception Error detected during operation.
     */
    template <typename T>
    inline auto parse(const std::string& value)
    -> typename std::enable_if<std::is_floating_point<T>::value, T>::type
    {
        return static_cast<T>(std::stold(value));
    }

    /**
     * Sets the variable with a new value and returns the old one.
     * @tparam T The target variable's type.
     * @tparam U The new value's type.
     * @param a The object whose value must be replaced.
     * @param b The value to assign to the object.
     * @return The previous value of the object.
     */
    template <typename T, typename U>
    __host__ __device__ inline constexpr auto exchange(T& a, U&& b) noexcept(
        std::is_nothrow_move_constructible<T>::value &&
        std::is_nothrow_assignable<T&, U>::value
    ) -> typename std::enable_if<std::is_convertible<U, T>::value, T>::type
    {
        T x = std::move(a);
          a = std::forward<U>(b);
        return x;
    }

    /**
     * Swaps the contents of two variables of same type
     * @tparam T The variables' type.
     * @param a The first variable to have its contents swapped.
     * @param b The second variable to have its contents swapped.
     */
    template <typename T>
    __host__ __device__ inline constexpr void swap(T& a, T& b) noexcept(
        std::is_nothrow_move_constructible<T>::value &&
        std::is_nothrow_move_assignable<T>::value
    ) {
        T x = std::move(a);
          a = std::move(b);
          b = std::move(x);
    }

    /**
     * Swaps the elements of two arrays of same type and size.
     * @tparam T The arrays' elements' swappable type.
     * @tparam N The arrays' size.
     * @param a The first array to have its elements swapped.
     * @param b The second array to have its elements swapped.
     */
    template <typename T, size_t N>
    __host__ __device__ inline constexpr void swap(T (&a)[N], T (&b)[N]) noexcept(noexcept(swap(*a, *b)))
    {
        for(size_t i = 0; i < N; ++i)
            swap(a[i], b[i]);
    }

    /**
     * Swallows a variadic list of parameters and returns the first one. This
     * functions is useful when dealing with type-packs.
     * @tparam T The type of the value to be returned.
     * @tparam U The type of the values to ignore.
     * @return The given return value.
     */
    template <typename T, typename ...U>
    __host__ __device__ inline constexpr T swallow(T&& target, U&&...) noexcept
    {
        return target;
    }

    /**
     * Invokes a functor that takes no arguments.
     * @tparam F The functor type.
     * @param lambda The functor instance to be invoked.
     * @return The functor invokation result.
     */
    template <typename F>
    __host__ __device__ inline constexpr decltype(auto) invoke(const F& lambda) {
        return (lambda)();
    }

    /**
     * Seamlessly invokes a functor depending on whether its a free function,
     * a lambda or an unbound member function pointer.
     * @tparam F The functor type.
     * @tparam O The first invoking parameter type or object type to bind to.
     * @tparam A The remaining invoking arguments types.
     * @param lambda The functor instance to be invoked.
     * @param object The possible object instance to bind the functor to.
     * @param args The remaining invoking arguments.
     * @return The functor invokation result.
     */
    template <typename F, typename O, typename ...A>
    __host__ __device__ inline constexpr decltype(auto) invoke(const F& lambda, O&& object, A&&... args)
    {
        if constexpr (std::is_member_function_pointer<F>::value) {
            return (object.*lambda)(std::forward<decltype(args)>(args)...);
        } else {
            return (lambda)(
                std::forward<decltype(object)>(object)
              , std::forward<decltype(args)>(args)...
            );
        }
    }
}

MUSEQA_END_NAMESPACE
