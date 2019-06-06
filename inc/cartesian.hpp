/** 
 * Multiple Sequence Alignment cartesian header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef CARTESIAN_HPP_INCLUDED
#define CARTESIAN_HPP_INCLUDED

#include <utility>

#include "utils.hpp"
#include "tuple.hpp"
#include "exception.hpp"
#include "reflection.hpp"

/**#@+
 * Represents a D-dimensional cartesian value. This value can then be expanded
 * to or viewed as point, a vector or ultimately a space formed from the origin.
 * @tparam D The number of cartesian dimensions.
 * @tparam T The type of dimension values.
 * @since 0.1.1
 */
template <size_t D, typename T = size_t>
struct Cartesian;

template <typename T>
struct Cartesian<1, T> : public Reflector
{
    static_assert(std::is_integral<T>::value, "cartesian dimensions must be integers!");

    T dim = {};                 /// The cartesian uni-dimensional value.

    __host__ __device__ inline constexpr Cartesian() noexcept = default;
    __host__ __device__ inline constexpr Cartesian(const Cartesian&) noexcept = default;
    __host__ __device__ inline constexpr Cartesian(Cartesian&&) noexcept = default;

    /**
     * Constructs a new uni-dimensional cartesian value instance.
     * @param value The cartesian value.
     */
    __host__ __device__ inline constexpr Cartesian (const T& value) noexcept
    :   dim {value}
    {}

    __host__ __device__ inline Cartesian& operator=(const Cartesian&) noexcept = default;
    __host__ __device__ inline Cartesian& operator=(Cartesian&&) noexcept = default;

    /**
     * Gives direct access to the a cartesian dimension value.
     * @param id The requested dimension identifier.
     * @return The dimension's value.
     */
    __host__ __device__ inline constexpr T operator[](ptrdiff_t id) const
    {
        enforce(static_cast<unsigned>(id) >= 1, "point dimension out of range");
        return dim;
    }

    /**
     * Allows a uni-dimensional cartesian value to be represented as a number.
     * @return The uni-dimensional cartesian value.
     */
    __host__ __device__ inline constexpr operator T() const noexcept
    {
        return dim;
    }

    /**
     * Calculates the total cartesian space volume.
     * @return The cartesian space volume.
     */
    __host__ __device__ inline constexpr T getVolume() const noexcept
    {
        return dim;
    }

    /**
     * Collapses a cartesian value into a 1-dimensional value using the current
     * value as the base. As we are already in the 1-dimension realm, there's
     * not much to do here besides returning the given value.
     * @param value The value to collpase to.
     * @return The collapsed cartesian value.
     */
    template <typename U>
    __host__ __device__ inline constexpr T collapseTo(const U& value) const noexcept
    {
        return static_cast<T>(value);
    }

    /**
     * Creates a new cartesian value instance from a tuple.
     * @tparam U The tuple type.
     * @param value The tuple from which the new cartesian will be created.
     * @return The new cartesian instance.
     */
    template <typename U>
    __host__ __device__ inline static constexpr Cartesian fromTuple(const Tuple<U>& value) noexcept
    {
        return {tuple::get<0>(value)};
    }

    using reflex = decltype(reflex(dim));
};

template <size_t D, typename T>
struct Cartesian : public Reflector
{
    static_assert(D > 0, "cartesian values are at least 1-dimensional!");
    static_assert(std::is_integral<T>::value, "cartesian dimensions must be integers!");

    T dim[D] = {};              /// The cartesian dimension values.

    __host__ __device__ inline constexpr Cartesian() noexcept = default;
    __host__ __device__ inline constexpr Cartesian(const Cartesian&) noexcept = default;
    __host__ __device__ inline constexpr Cartesian(Cartesian&&) noexcept = default;

    /**
     * Constructs a new cartesian value instance.
     * @param value The cartesian dimensional values.
     */
    template <typename ...U, typename X = T, typename = typename std::enable_if<
            utils::all(std::is_convertible<U, X>::value...)
        >::type >
    __host__ __device__ inline constexpr Cartesian(U&&... value) noexcept
    :   dim {static_cast<T>(value)...}
    {}

    __host__ __device__ inline Cartesian& operator=(const Cartesian&) noexcept = default;
    __host__ __device__ inline Cartesian& operator=(Cartesian&&) noexcept = default;

    /**
     * Gives direct access to the a cartesian dimension value.
     * @param id The requested dimension identifier.
     * @return The dimension's value.
     */
    __host__ __device__ inline constexpr T operator[](ptrdiff_t id) const
    {
        enforce(static_cast<unsigned>(id) >= D, "point dimension out of range");
        return dim[id];
    }

    /**
     * The addition of two cartesians values.
     * @param value The cartesian value to add to the current one.
     * @return The new cartesian value.
     */
    template <typename U>
    __host__ __device__ inline constexpr Cartesian operator+(const Cartesian<D, U>& value) const noexcept
    {
        using namespace tuple;
        using namespace utils;
        return fromTuple(zipWith(add<T>, tie(dim), tie(value.dim)));
    }

    /**
     * The scalar product of a cartesian value.
     * @param scalar The scalar to multiply the cartesian value by.
     * @return The new cartesian value.
     */
    template <typename U>
    __host__ __device__ inline constexpr Cartesian operator*(const U& scalar) const noexcept
    {
        using namespace tuple;
        using namespace utils;
        return fromTuple(apply(mul<T>, static_cast<T>(scalar), tie(dim)));
    }

    /**
     * Calculates the total cartesian space volume.
     * @return The cartesian space volume.
     */
    __host__ __device__ inline constexpr T getVolume() const noexcept
    {
        using namespace tuple;
        using namespace utils;
        return foldl(mul<T>, T {1}, tie(dim));
    }

    /**
     * Collapses a multidimensional cartesian value into 1-dimensional value
     * using this value as base. This is useful for accessing buffers.
     * @param value The cartesian value to collapsed to.
     * @return The 1-dimensional collapsed cartesian value.
     */
    template <typename U = T>
    __host__ __device__ inline constexpr T collapseTo(const Cartesian<D, U>& value) const noexcept
    {
        using namespace tuple;
        using namespace utils;
        return foldl(add<T>, 0, zipWith(mul<T>, tie(value.dim), scanr(mul<T>, 1, tail(tie(dim)))));
    }

    /**
     * Creates a new cartesian value instance from a tuple.
     * @tparam I The sequence of tuple indeces.
     * @tparam U The tuple types.
     * @param value The tuple from which the new cartesian will be created.
     * @return The new cartesian instance.
     */
    template <size_t ...I, typename ...U>
    __host__ __device__ inline static constexpr Cartesian fromTuple
        (   const tuple::BaseTuple<Indexer<I...>, U...>& value      )
        noexcept
    {
        return {tuple::get<I>(value)...};
    }

    using reflex = decltype(reflect(dim));
};
/**#@-*/

/**
 * Compares two cartesian values of same dimensionality.
 * @param a The first value to compare.
 * @param b The second value to compare.
 */
template <size_t D, typename T, typename U>
__host__ __device__ inline constexpr bool operator>(const Cartesian<D, T>& a, const Cartesian<D, U>& b) noexcept
{
    using namespace tuple;
    using namespace utils;
    return foldl(andl, true, zipWith(gt<T>, tie(a.dim), tie(b.dim)));
}

/**
 * Compares two cartesian values of same dimensionality.
 * @param a The first value to compare.
 * @param b The second value to compare.
 */
template <size_t D, typename T, typename U>
__host__ __device__ inline constexpr bool operator>=(const Cartesian<D, T>& a, const Cartesian<D, U>& b) noexcept
{
    using namespace tuple;
    using namespace utils;
    return foldl(andl, true, zipWith(gte<T>, tie(a.dim), tie(b.dim)));
}

/**
 * Compares two cartesian values of same dimensionality.
 * @param a The first value to compare.
 * @param b The second value to compare.
 */
template <size_t D, typename T, typename U>
__host__ __device__ inline constexpr bool operator<(const Cartesian<D, T>& a, const Cartesian<D, U>& b) noexcept
{
    return b > a;
}

/**
 * Compares two cartesian values of same dimensionality.
 * @param a The first value to compare.
 * @param b The second value to compare.
 */
template <size_t D, typename T, typename U>
__host__ __device__ inline constexpr bool operator<=(const Cartesian<D, T>& a, const Cartesian<D, U>& b) noexcept
{
    return b >= a;
}

/**
 * Checks whether two cartesian values are equal.
 * @tparam D The dimensionality of both values.
 * @param a The first value to compare.
 * @param b The second value to compare.
 */
template <size_t D, typename T, typename U>
__host__ __device__ inline constexpr bool operator==(const Cartesian<D, T>& a, const Cartesian<D, U>& b) noexcept
{
    using namespace tuple;
    using namespace utils;
    return foldl(andl, true, zipWith(eq<T>, tie(a.dim), tie(b.dim)));
}

/**
 * Checks whether two cartesian values are equal.
 * @tparam D The dimensionality of first value.
 * @tparam E The dimensionality of second value.
 */
template <size_t D, size_t E, typename T, typename U>
__host__ __device__ inline constexpr bool operator==(const Cartesian<D, T>&, const Cartesian<E, U>&) noexcept
{
    return false;
}

/**
 * Checks whether two cartesian values are different.
 * @tparam D The dimensionality of first value.
 * @tparam E The dimensionality of second value.
 * @param a The first value to compare.
 * @param b The second value to compare.
 */
template <size_t D, size_t E, typename T, typename U>
__host__ __device__ inline bool constexpr operator!=(const Cartesian<D, T>& a, const Cartesian<E, U>& b) noexcept
{
    return !(a == b);
}

#endif