/** 
 * Multiple Sequence Alignment optional type header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef OPTIONAL_HPP_INCLUDED
#define OPTIONAL_HPP_INCLUDED

#pragma once

#include <cstddef>
#include <utility>

#include "utils.hpp"

namespace detail
{
    /*
     * Forward declaration of Maybe, so it can be used as a complete type
     * before defining it properly.
     */
    template <typename T>
    struct Maybe;

    /**
     * Represents the "nothing" value type. This struct is empty and shall always
     * represent a failed or empty value.
     * @since 0.1.1
     */
    template <>
    struct Maybe<void>
    {
        /**
         * Checks whether the object is equal to another one.
         * @tparam The value-type of other object.
         * @param (ignored) The other object to compare.
         * @return Are the objects equal?
         */
        template <typename U>
        inline constexpr bool operator==(const Maybe<U>&)
        {
            return std::is_same<U, void>::value;
        }

        /**
         * Checks whether the object is different of another one.
         * @tparam The value-type of other object.
         * @param (ignored) The other object to compare.
         * @return Are the objects different?
         */
        template <typename U>
        inline constexpr bool operator!=(const Maybe<U>&)
        {
            return !std::is_same<U, void>::value;
        }
    };

    /**
     * Heavily inspired by Haskell, this struct represents an optional value.
     * It may contain a value of type T or nothing.
     * @tparam T The type of optional value.
     * @since 0.1.1
     */
    template <typename T>
    struct Maybe
    {
        T value;                /// The value that may be held by the object.
        bool isEmpty = true;    /// Is the object actually empty?

        inline constexpr Maybe() = default;

        /**
         * Builds a new valued-object.
         * @param value The value to be held by the object.
         */
        inline constexpr Maybe(const T& value)
        : value {value}
        , isEmpty(false) {}

        /**
         * Builds a new valued-object.
         * @param value The value to be moved to the object.
         */
        inline constexpr Maybe(T&& value)
        : value {std::move(value)}
        , isEmpty(false) {}

        /**
         * Builds a new empty object.
         * @param (ignored) The invalid object.
         */
        inline constexpr Maybe(const Maybe<void>&)
        : isEmpty(true) {}

        /**
         * Copies an optional object.
         * @param other The object to be copied.
         */
        inline constexpr Maybe(const Maybe<T>& other)
        : isEmpty(other.isEmpty)
        {
            if(!isEmpty) {
                new (&value) T(other.value);
            }
        }

        /**
         * Copies an optional object.
         * @param other The object to be copied.
         */
        inline constexpr Maybe(Maybe<T>&& other)
        : isEmpty(other.isEmpty)
        {
            if(!isEmpty) {
                new (&value) T(std::move(other.value));
            }
        }

        /**
         * Checks whether the object is equal to another one.
         * @tparam U The value-type of other object.
         * @param other The other object to compare.
         * @return Are the objects equal?
         */
        template <typename U>
        inline constexpr bool operator==(const Maybe<U>& other) const
        {
            return isEmpty == other.isEmpty
                ? !isEmpty && utils::Comparable<T, U>::value && (value == other.value)
                : false;
        }

        /**
         * Checks whether the object is different from another one.
         * @tparam The value-type of other object.
         * @param other The other object to compare.
         * @return Are the objects different?
         */
        template <typename U>
        inline constexpr bool operator!=(const Maybe<U>& other) const
        {
            return !(*this == other);
        }

        /**
         * Converts the object into a bool, informing whether empty.
         * @return Is the object empty?
         */
        inline constexpr operator bool() const
        {
            return !empty();
        }

        /**
         * Gets the value held by the object.
         * @return The value within the object.
         */
        inline constexpr const T& get(const T& fallback = {}) const
        {
            return !isEmpty ? value : fallback;
        }

        /**
         * Informs whether the object is empty or not.
         * @return Is the object empty?
         */
        inline constexpr bool empty() const noexcept
        {
            return isEmpty;
        }
    };
};

/**
 * The optional type object. This represents a type that may not be initialized.
 * @since 0.1.1
 */
template <typename T>
using Maybe = detail::Maybe<utils::Pure<T>>;

/**
 * Aliasing for the always empty type.
 * @since 0.1.1
 */
using Nothing = Maybe<void>;

/**#@+
 * Creates a new value wrapped in optional context.
 * @tparam T The value type.
 * @param value The value to be wrapped.
 * @return The wrapped value.
 */
template <typename T>
inline auto maybe(const T& value) -> Maybe<T>
{
    return {value};
}

template <typename T>
auto maybe(T&& value) -> Maybe<T>
{
    return {std::move(value)};
}
/**#@-*/

/**
 * Creates a new optional context with no value.
 * @tparam T The type of expected value.
 * @return The empty context.
 */
template <typename T = void>
inline auto nothing() -> Maybe<T>
{
    return {};
}

namespace detail
{
    /**
     * The left type value for an @ref Either instance. This type usually
     * carries an error or denotes success was not achieved.
     * @tparam T The value type.
     * @since 0.1.1
     */
    template <typename T>
    struct Left
    {
        T value;    /// The value held by instance.
    };

    /**
     * The right type value for an @ref Either instance. This type usually
     * carries a well-successed value or denotes success was achieved.
     * @tparam T The value type.
     * @since 0.1.1
     */
    template <typename T>
    struct Right
    {
        T value;    /// The value held by instance.
    };

    /*
     * Forward declaration of Either, so it can be used as a complete type
     * before defining it properly.
     */
    template <typename L, typename R>
    struct Either;

    /**
     * Heavilly inspired by Haskell, this struct represents a value whose type
     * is in between two alternatives. This is usually useful to return errors
     * instead of throwing expensive exceptions.
     * @param L The left type alternative.
     * @param R The right type alternative.
     * @since 0.1.1
     */
    template <typename L, typename R>
    struct Either
    {
        /**
         * This object holds a value whose value is an alternative.
         * @see Maybe
         * @since 0.1.1
         */
        union
        {
            L leftValue;        /// The left-type value reference.
            R rightValue;       /// The right-type value reference.
        };

        bool isLeft;            /// Is left the current value type?

        Either() = delete;

        /**
         * Builds a new instance using left-type option.
         * @param l The left-typed value.
         */
        inline constexpr Either(const Left<L>& l)
        : leftValue {l.value}
        , isLeft(true) {}

        /**
         * Builds a new instance using right-type option.
         * @param r The right-typed value.
         */
        inline constexpr Either(const Right<R>& r)
        : rightValue {r.value}
        , isLeft(false) {}

        /**
         * Builds a new instance using left-type option.
         * @param l The left-typed value.
         */
        inline Either(Left<L>&& l)
        : leftValue {std::move(l.value)}
        , isLeft(true) {}

        /**
         * Builds a new instance using right-type option.
         * @param r The right-typed value.
         */
        inline Either(Right<R>&& r)
        : rightValue {std::move(r.value)}
        , isLeft(false) {}

        /**
         * Copies an alternative-typed object.
         * @param other The object to be copied.
         */
        inline constexpr Either(const Either<L, R>& other)
        : isLeft(other.isLeft)
        {
            isLeft
                ? new (&leftValue) L(other.leftValue)
                : new (&rightValue) R(other.rightValue);
        }

        /**
         * Copies an alternative-typed object.
         * @param other The object to be copied.
         */
        inline constexpr Either(Either<L, R>&& other)
        : isLeft(other.isLeft)
        {
            isLeft
                ? new (&leftValue) L(std::move(other.leftValue))
                : new (&rightValue) R(std::move(other.rightValue));
        }

        /**
         * Calls the correct type-destructor of value.
         * @see Either
         */
        inline ~Either() noexcept
        {
            isLeft
                ? leftValue.~L()
                : rightValue.~R();
        }

        /**
         * Checks whether the object is equal to another one.
         * @tparam X The left value-type of other object.
         * @tparam Y The right value-type of other object.
         * @param other The other object to compare.
         * @return Are the objects equal?
         */
        template <typename X, typename Y>
        inline constexpr bool operator==(const Either<X, Y>& other)
        {
            return isLeft == other.isLeft
                ?  isLeft && utils::Comparable<L, X>::value && left() == other.left()
                : !isLeft && utils::Comparable<R, Y>::value && right() == other.right();
        }

        /**
         * Checks whether the object is different from another one.
         * @tparam X The left value-type of other object.
         * @tparam Y The right value-type of other object.
         * @param other The other object to compare.
         * @return Are the objects different?
         */
        template <typename X, typename Y>
        inline bool operator!=(const Either<X, Y>& other)
        {
            return !(*this == other);
        }

        /**
         * Converts to a bool. Right values are taken as true values, whereas left
         * are used for errors, and thus represent a false value.
         * @return The object's bool value.
         */
        inline constexpr operator bool() const
        {
            return !isLeft;
        }

        /**
         * Gets the left value stored by the object if the value stored is left-typed.
         * @return The left value of object.
         */
        inline constexpr auto left() const -> Maybe<L>
        {
            return isLeft
                ? maybe(leftValue)
                : nothing();
        }

        /**
         * Gets the right value stored by the object if the value stored is right-typed.
         * @return The right value of object.
         */
        inline constexpr auto right() const -> Maybe<R>
        {
            return isLeft
                ? nothing()
                : maybe(rightValue);
        }
    };
};

template <typename T>
using Left = detail::Left<utils::Pure<T>>;

template <typename T>
using Right = detail::Right<utils::Pure<T>>;

/**
 * The alternative typed object. This object allows holding a value whose type
 * is one of two options.
 * @tparam L The left type option.
 * @tparam R The right type option.
 * @since 0.1.1
 */
template <typename L, typename R>
using Either = detail::Either<utils::Pure<L>, utils::Pure<R>>;

/**#@+
 * Creates a new value wrapped in a left alternative context.
 * @tparam T The value type.
 * @param value The value to be wrapped.
 * @return The wrapped value.
 */
template <typename T>
constexpr auto left(const T& x) -> Left<T>
{
    return {x};
}

template <typename T>
auto left(T&& x) -> Left<T>
{
    return {std::move(x)};
}
/**#@-*/

/**#@+
 * Creates a new value wrapped in a right alternative context.
 * @tparam T The value type.
 * @param value The value to be wrapped.
 * @return The wrapped value.
 */
template <typename T>
constexpr auto right(const T& x) -> Right<T>
{
    return {x};
}

template <typename T>
auto right(T&& x) -> Right<T>
{
    return {std::move(x)};
}
/**#@-*/

#endif