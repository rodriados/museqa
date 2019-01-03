/** 
 * Multiple Sequence Alignment expect type header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef EXPECT_HPP_INCLUDED
#define EXPECT_HPP_INCLUDED

#include <cstddef>
#include <utility>

namespace expect
{
    /**
     * Checks whether two types can be compared to each other.
     * @tparam A First type to check.
     * @tparam B Second type to check.
     * @return Are the types comparable.
     */
    template <typename A, typename B>
    inline constexpr bool comparable()
    {
        return std::is_same<A, B>::value
            || (std::is_arithmetic<A>::value && std::is_arithmetic<B>::value)
            || std::is_convertible<B, A>::value;
    }

    /*
     * Forward declaration of Expect, so it can be used as a complete type
     * before defining it properly.
     */
    template <typename T>
    struct Expect;

    /**
     * Represents the "nothing" value type. This struct is empty and shall always
     * represent a failed or empty value.
     * @since 0.1.1
     */
    template <>
    struct Expect<void>
    {
        /**
         * Checks whether the object is equal to another one.
         * @tparam The value-type of other object.
         * @param (ignored) The other object to compare.
         * @return Are the objects equal?
         */
        template <typename U>
        inline bool operator==(const Expect<U>&) const
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
        inline bool operator!=(const Expect<U>&) const
        {
            return !std::is_same<U, void>::value;
        }

        /**
         * Informs how the object corresponds to a condition.
         * @return Does object fulfill expectancy?
         */
        inline operator bool() const
        {
            return false;
        }
    };

    /**
     * Heavily inspired by Haskell, this struct represents an optional value.
     * It may contain a value of type T or nothing.
     * @tparam T The type of optional value.
     * @since 0.1.1
     */
    template <typename T>
    struct Expect
    {
        T value;                /// The value that may be held by the object.
        bool isEmpty = true;    /// Is the object actually empty?

        Expect() = default;

        /**
         * Builds a new valued-object.
         * @tparam X The value type.
         * @param value The value to be held by the object.
         */
        template <typename X = T>
        inline Expect(const X& value)
        : value {static_cast<T>(value)}
        , isEmpty(false) {}

        /**
         * Builds a new valued-object.
         * @param value The value to be moved to the object.
         */
        inline Expect(T&& value)
        : value {std::move(value)}
        , isEmpty(false) {}

        /**
         * Builds a new empty object.
         * @param (ignored) The invalid object.
         */
        inline Expect(const Expect<void>&)
        : isEmpty(true) {}

        /**
         * Copies an optional object.
         * @tparam X The value type.
         * @param other The object to be copied.
         */
        template <typename X = T>
        inline Expect(const Expect<X>& other)
        : isEmpty(other.isEmpty)
        {
            if(!isEmpty) {
                new (&value) T(static_cast<T>(other.value));
            }
        }

        /**
         * Copies an optional object.
         * @param other The object to be copied.
         */
        inline Expect(Expect<T>&& other)
        : isEmpty(other.isEmpty)
        {
            if(!isEmpty) {
                new (&value) T(std::move(other.value));
            }
        }

        /**
         * Checks whether the object is equal to another one.
         * @tparam X The value-type of other object.
         * @param other The other object to compare.
         * @return Are the objects equal?
         */
        template <typename X = T>
        inline bool operator==(const Expect<X>& other) const
        {
            return isEmpty == other.isEmpty
                ? !isEmpty && comparable<T, X>() && (value == other.value)
                : false;
        }

        /**
         * Checks whether the object is different from another one.
         * @tparam The value-type of other object.
         * @param other The other object to compare.
         * @return Are the objects different?
         */
        template <typename X = T>
        inline bool operator!=(const Expect<X>& other) const
        {
            return !(*this == other);
        }

        /**
         * Converts the object into a bool, informing whether empty.
         * @return Is the object empty?
         */
        inline operator bool() const
        {
            return !empty();
        }

        /**
         * Gets the value held by the object.
         * @return The value within the object.
         */
        inline const T& get(const T& fallback = {}) const
        {
            return !isEmpty ? value : fallback;
        }

        /**
         * Informs whether the object is empty or not.
         * @return Is the object empty?
         */
        inline bool empty() const noexcept
        {
            return isEmpty;
        }
    };
};

/**
 * The expected type object. This represents an expected type that may not be initialized.
 * @since 0.1.1
 */
template <typename T>
using Expect = expect::Expect<T>;

/**
 * Aliasing for the always empty type.
 * @since 0.1.1
 */
using Nothing = expect::Expect<void>;

/**
 * Creates a new value wrapped in an expected context.
 * @tparam T The value type.
 * @param value The value to be wrapped.
 * @return The wrapped value.
 */
template <typename T>
inline auto expected(const T& value) -> Expect<T>
{
    return {value};
}

/**
 * Creates a new value wrapped in expected context.
 * @tparam T The value type.
 * @param value The value to be wrapped.
 * @return The wrapped value.
 */
template <typename T>
inline auto expected(T&& value) -> Expect<T>
{
    return {std::move(value)};
}

/**
 * Creates a new optional context with no value.
 * @tparam T The type of expected value.
 * @return The empty context.
 */
template <typename T = void>
inline auto nothing() -> Expect<T>
{
    return {};
}

namespace expect
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
         * @see Expect
         * @since 0.1.1
         */
        union
        {
            L leftValue;        /// The left-type value reference.
            R rightValue;       /// The right-type value reference.
        };

        bool isLeft = false;    /// Is the currently active value type left?

        Either() = delete;

        /**
         * Builds a new instance using left-type option.
         * @tparam X The left-type.
         * @param l The left-typed value.
         */
        template <typename X = L>
        inline Either(const Left<X>& l)
        : leftValue {static_cast<L>(l.value)}
        , isLeft(true) {}

        /**
         * Builds a new instance using right-type option.
         * @tparam X The right-type.
         * @param r The right-typed value.
         */
        template <typename X = R>
        inline Either(const Right<X>& r)
        : rightValue {static_cast<R>(r.value)}
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
         * @tparam X The left value-type of other object.
         * @tparam Y The right value-type of other object.
         * @param other The object to be copied.
         */
        template <typename X, typename Y>
        inline Either(const Either<X, Y>& other)
        : isLeft(other.isLeft)
        {
            static_assert(std::is_convertible<X, L>::value, "Cannot convert left types.");
            static_assert(std::is_convertible<Y, R>::value, "Cannot convert right types.");

            isLeft == true
                ? (void *) new (&leftValue) L(other.leftValue)
                : (void *) new (&rightValue) R(other.rightValue);
        }

        /**
         * Copies an alternative-typed object.
         * @param other The object to be copied.
         */
        inline Either(Either<L, R>&& other)
        : isLeft(other.isLeft)
        {
            isLeft == true
                ? (void *) new (&leftValue) L(std::move(other.leftValue))
                : (void *) new (&rightValue) R(std::move(other.rightValue));
        }

        /**
         * Calls the correct type-destructor of value.
         * @see Either
         */
        inline ~Either() noexcept
        {
            isLeft == true
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
        inline bool operator==(const Either<X, Y>& other)
        {
            return isLeft == other.isLeft
                ? isLeft == true  && comparable<L, X>() && left() == other.left()
                : isLeft == false && comparable<R, Y>() && right() == other.right();
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
        inline operator bool() const
        {
            return !isLeft;
        }

        /**
         * Gets the left value stored by the object if the value stored is left-typed.
         * @return The left value of object.
         */
        inline auto left() const -> Expect<L>
        {
            return isLeft == true
                ? expected(leftValue)
                : nothing();
        }

        /**
         * Gets the right value stored by the object if the value stored is right-typed.
         * @return The right value of object.
         */
        inline auto right() const -> Expect<R>
        {
            return isLeft == false
                ? expected(rightValue)
                : nothing();
        }
    };
};

/**
 * Exposing the left-type wrapper.
 * @tparam T The wrapped type.
 * @since 0.1.1
 */
template <typename T>
using Left = expect::Left<T>;

/**
 * Exposing the right-type wrapper.
 * @tparam T The wrapped type.
 * @since 0.1.1
 */
template <typename T>
using Right = expect::Right<T>;

/**
 * The alternative typed object. This object allows holding a value whose type
 * is one of two options.
 * @tparam L The left type option.
 * @tparam R The right type option.
 * @since 0.1.1
 */
template <typename L, typename R>
using Either = expect::Either<L, R>;

/**
 * Creates a new value wrapped in a left alternative context.
 * @tparam T The value type.
 * @param value The value to be wrapped.
 * @return The wrapped value.
 */
template <typename T>
inline auto left(const T& x) -> Left<T>
{
    return {x};
}

/**
 * Creates a new value wrapped in a left alternative context.
 * @tparam T The value type.
 * @param value The value to be wrapped.
 * @return The wrapped value.
 */
template <typename T>
inline auto left(T&& x) -> Left<T>
{
    return {std::move(x)};
}

/**
 * Creates a new value wrapped in a right alternative context.
 * @tparam T The value type.
 * @param value The value to be wrapped.
 * @return The wrapped value.
 */
template <typename T>
inline auto right(const T& x) -> Right<T>
{
    return {x};
}

/**
 * Creates a new value wrapped in a right alternative context.
 * @tparam T The value type.
 * @param value The value to be wrapped.
 * @return The wrapped value.
 */
template <typename T>
inline auto right(T&& x) -> Right<T>
{
    return {std::move(x)};
}

#endif