/** 
 * Multiple Sequence Alignment cartesian header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef CARTESIAN_HPP_INCLUDED
#define CARTESIAN_HPP_INCLUDED

#include <utility>

#include <utils.hpp>
#include <tuple.hpp>
#include <exception.hpp>
#include <reflection.hpp>

/**#@+
 * Represents a D-dimensional cartesian value. This value can then be expanded
 * to or viewed as point, a vector or ultimately a space formed from the origin.
 * @tparam D The number of cartesian dimensions.
 * @tparam T The type of dimension values.
 * @since 0.1.1
 */
template <size_t D, typename T = ptrdiff_t>
struct cartesian;

template <typename T>
struct cartesian<1, T> : public reflector
{
    static_assert(std::is_integral<T>::value, "cartesian dimensions must be integers");

    using element_type = T;                     /// The cartesian dimension type.
    static constexpr size_t dimensionality = 1; /// The cartesian dimensionality.

    element_type dim = {};                      /// The cartesian uni-dimensional value.

    __host__ __device__ inline constexpr cartesian() noexcept = default;
    __host__ __device__ inline constexpr cartesian(const cartesian&) noexcept = default;
    __host__ __device__ inline constexpr cartesian(cartesian&&) noexcept = default;

    /**
     * Constructs a new uni-dimensional cartesian value instance.
     * @param value The cartesian value.
     */
    __host__ __device__ inline constexpr cartesian (const element_type& value) noexcept
    :   dim {value}
    {}

    __host__ __device__ inline cartesian& operator=(const cartesian&) noexcept = default;
    __host__ __device__ inline cartesian& operator=(cartesian&&) noexcept = default;

    /**
     * Gives direct access to the a cartesian dimension value.
     * @param id The requested dimension identifier.
     * @return The dimension's value.
     */
    __host__ __device__ inline element_type operator[](ptrdiff_t id) const
    {
        enforce(size_t(id) < 1, "point dimension out of range");
        return dim;
    }

    /**
     * Allows a uni-dimensional cartesian value to be represented as a number.
     * @return The uni-dimensional cartesian value.
     */
    __host__ __device__ inline constexpr operator element_type() const noexcept
    {
        return dim;
    }

    /**
     * Calculates the total cartesian space volume.
     * @return The cartesian space volume.
     */
    __host__ __device__ inline constexpr size_t volume() const noexcept
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
    __host__ __device__ inline constexpr element_type collapse(const U& value) const noexcept
    {
        return static_cast<element_type>(value);
    }

    /**
     * Creates a new cartesian value instance from a tuple.
     * @tparam U The tuple type.
     * @param value The tuple from which the new cartesian will be created.
     * @return The new cartesian instance.
     */
    template <typename U>
    __host__ __device__ inline static constexpr cartesian from_tuple(const tuple<U>& value) noexcept
    {
        return {internal::tuple::get<0>(value)};
    }

    using reflex = decltype(reflex(dim));
};

template <size_t D, typename T>
struct cartesian : public reflector
{
    static_assert(D > 0, "cartesian values are at least 1-dimensional");
    static_assert(std::is_integral<T>::value, "cartesian dimensions must be integers");

    using element_type = T;                     /// The cartesian dimension type.
    static constexpr size_t dimensionality = D; /// The cartesian dimensionality.

    element_type dim[D] = {};                   /// The cartesian dimension values.

    __host__ __device__ inline constexpr cartesian() noexcept = default;
    __host__ __device__ inline constexpr cartesian(const cartesian&) noexcept = default;
    __host__ __device__ inline constexpr cartesian(cartesian&&) noexcept = default;

    /**
     * Constructs a new cartesian value instance.
     * @param value The cartesian dimensional values.
     */
    template <typename ...U, typename X = T, typename = typename std::enable_if<
            utils::all(std::is_convertible<U, X>::value...)
        >::type >
    __host__ __device__ inline constexpr cartesian(U&&... value) noexcept
    :   dim {static_cast<element_type>(value)...}
    {}

    __host__ __device__ inline cartesian& operator=(const cartesian&) noexcept = default;
    __host__ __device__ inline cartesian& operator=(cartesian&&) noexcept = default;

    /**
     * Gives direct access to the a cartesian dimension value.
     * @param id The requested dimension identifier.
     * @return The dimension's value.
     */
    __host__ __device__ inline element_type operator[](ptrdiff_t id) const
    {
        enforce(size_t(id) < D, "point dimension out of range");
        return dim[id];
    }

    /**
     * The addition of two cartesians values.
     * @param value The cartesian value to add to the current one.
     * @return The new cartesian value.
     */
    template <typename U>
    __host__ __device__ inline constexpr cartesian operator+(const cartesian<D, U>& value) const noexcept
    {
        using namespace utils;
        return from_tuple(zipwith(add<element_type>, tie(dim), tie(value.dim)));
    }

    /**
     * The scalar product of a cartesian value.
     * @param scalar The scalar to multiply the cartesian value by.
     * @return The new cartesian value.
     */
    template <typename U>
    __host__ __device__ inline constexpr cartesian operator*(const U& scalar) const noexcept
    {
        using namespace utils;
        return from_tuple(apply(mul<element_type>, static_cast<element_type>(scalar), tie(dim)));
    }

    /**
     * Calculates the total cartesian space volume.
     * @return The cartesian space volume.
     */
    __host__ __device__ inline constexpr size_t volume() const noexcept
    {
        using namespace utils;
        return foldl(mul<element_type>, element_type {1}, tie(dim));
    }

    /**
     * Collapses a multidimensional cartesian value into 1-dimensional value
     * using this value as base. This is useful for accessing buffers.
     * @param value The cartesian value to collapsed to.
     * @return The 1-dimensional collapsed cartesian value.
     */
    template <typename U = T>
    __host__ __device__ inline constexpr element_type collapse(const cartesian<D, U>& value) const noexcept
    {
        using namespace utils;
        return foldl(add<element_type>, 0, zipwith(
            mul<element_type>, tie(value.dim), scanr(mul<element_type>, 1, tail(tie(dim)))
        ));
    }

    /**
     * Creates a new cartesian value instance from a tuple.
     * @tparam I The sequence of tuple indeces.
     * @tparam U The tuple types.
     * @param value The tuple from which the new cartesian will be created.
     * @return The new cartesian instance.
     */
    template <size_t ...I, typename ...U>
    __host__ __device__ inline static constexpr cartesian from_tuple(
            const internal::tuple::base<indexer<I...>, U...>& value
        )
        noexcept
    {
        return {internal::tuple::get<I>(value)...};
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
__host__ __device__ inline constexpr bool operator>(const cartesian<D, T>& a, const cartesian<D, U>& b) noexcept
{
    using namespace utils;
    return foldl(andl, true, zipwith(gt<T>, tie(a.dim), tie(b.dim)));
}

/**
 * Compares two cartesian values of same dimensionality.
 * @param a The first value to compare.
 * @param b The second value to compare.
 */
template <size_t D, typename T, typename U>
__host__ __device__ inline constexpr bool operator>=(const cartesian<D, T>& a, const cartesian<D, U>& b) noexcept
{
    using namespace utils;
    return foldl(andl, true, zipwith(gte<T>, tie(a.dim), tie(b.dim)));
}

/**
 * Compares two cartesian values of same dimensionality.
 * @param a The first value to compare.
 * @param b The second value to compare.
 */
template <size_t D, typename T, typename U>
__host__ __device__ inline constexpr bool operator<(const cartesian<D, T>& a, const cartesian<D, U>& b) noexcept
{
    return b > a;
}

/**
 * Compares two cartesian values of same dimensionality.
 * @param a The first value to compare.
 * @param b The second value to compare.
 */
template <size_t D, typename T, typename U>
__host__ __device__ inline constexpr bool operator<=(const cartesian<D, T>& a, const cartesian<D, U>& b) noexcept
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
__host__ __device__ inline constexpr bool operator==(const cartesian<D, T>& a, const cartesian<D, U>& b) noexcept
{
    using namespace utils;
    return foldl(andl, true, zipwith(eq<T>, tie(a.dim), tie(b.dim)));
}

/**
 * Checks whether two cartesian values are equal.
 * @tparam D The dimensionality of first value.
 * @tparam E The dimensionality of second value.
 */
template <size_t D, size_t E, typename T, typename U>
__host__ __device__ inline constexpr bool operator==(const cartesian<D, T>&, const cartesian<E, U>&) noexcept
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
__host__ __device__ inline bool constexpr operator!=(const cartesian<D, T>& a, const cartesian<E, U>& b) noexcept
{
    return !(a == b);
}

#endif