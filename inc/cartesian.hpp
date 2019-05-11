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

/**
 * Represents a D-dimensional cartesian space or point.
 * @tparam D The number of cartesian dimensions.
 * @tparam T The type of dimension values.
 * @since 0.1.1
 */
template <size_t D, typename T = size_t>
struct Cartesian : public Reflector
{
    static_assert(D > 0, "cartesian spaces are at least 1-dimensional!");
    static_assert(std::is_integral<T>::value, "the cartesian dimensions must be integers!");

    const T dim[D] = {};                            /// The cartesian dimension values.
    static constexpr size_t dimensionality = D;     /// The cartesian dimensionality.

    __host__ __device__ inline constexpr Cartesian() noexcept = default;
    __host__ __device__ inline constexpr Cartesian(const Cartesian&) noexcept = default;
    __host__ __device__ inline constexpr Cartesian(Cartesian&&) noexcept = default;

    /**
     * Constructs a new cartesian space instance.
     * @param value The space's dimensional limits.
     */
    template <typename ...U>
    __host__ __device__ inline constexpr Cartesian(U&&... value) noexcept
    :   dim {static_cast<T>(value)...}
    {}

    __host__ __device__ inline Cartesian& operator=(const Cartesian&) noexcept = default;
    __host__ __device__ inline Cartesian& operator=(Cartesian&&) noexcept = default;

    /**
     * Gives direct access to the a cartesian dimensional value.
     * @param id The requested dimension identifier.
     * @return The dimension's value.
     */
    __host__ __device__ inline constexpr T operator[](ptrdiff_t id) const
    {
#if defined(msa_compile_cython) && !defined(msa_compile_cuda)
        if(static_cast<unsigned>(id) >= D)
            throw Exception("point dimension out of range");
#endif
        return dim[id];
    }

    /**
     * Allows a uni-dimensional cartesian value to be represented as a number.
     * @return The uni-dimensional cartesian value.
     */
    template <size_t I = D, typename = typename std::enable_if<(I == 1)>::type>
    __host__ __device__ inline constexpr operator T() const noexcept
    {
        return dim[0];
    }

    /**
     * Calculates the total cartesian space volume.
     * @return The cartesian space volume.
     */
    __host__ __device__ inline constexpr T getVolume() const noexcept
    {
        return tuple::foldl(utils::mul, T {1}, TupleN<T, D> {dim});
    }

    /**
     * Collapses a multidimensional cartesian point into a 1-dimensional value
     * using this space as base. This is useful for accessing buffers.
     * @param point The cartesian point to be collapsed.
     * @return The 1-dimensional cartesian value.
     */
    __host__ __device__ inline constexpr T collapseTo(const Cartesian& point) const noexcept
    {
        using namespace tuple;
        using namespace utils;
        using P = TupleN<T, D>;
        return foldl(add, 0, zipWith(mul, P {point.dim}, scanr(mul, 1, tail(P {dim}))));
    }

    /**
     * Checks whether a cartesian space contains a point within its limits.
     * @param point The point to be checked.
     * @return Is the point within the space's limits?
     */
    __host__ __device__ inline constexpr bool contains(const Cartesian& point) const noexcept
    {
        using namespace tuple;
        using namespace utils;
        using P = TupleN<T, D>;
        return foldl(andl, true, zipWith(lt, P {point.dim}, P {dim}));
    }

    using reflex = decltype(reflect(dim));
};

/**
 * Checks whether two cartesian values are equal.
 * @tparam D The dimensionality of both values.
 * @tparam T The first value entry type.
 * @tparam U The second value entry type.
 * @param a The first value to compare.
 * @param b The second value to compare.
 * @return Are both values equal?
 */
template <size_t D, typename T, typename U>
__host__ __device__ inline constexpr bool operator==(const Cartesian<D, T>& a, const Cartesian<D, U>& b) noexcept
{
    using P = TupleN<T, D>;
    return tuple::foldl(utils::andl, true, tuple::zipWith(utils::eq, P {a.dim}, P {b.dim}));
}

/**
 * Checks whether two cartesian values are equal.
 * @tparam D The dimensionality of first value.
 * @tparam E The dimensionality of second value.
 * @tparam T The first value entry type.
 * @tparam U The second value entry type.
 * @return With different dimensionalities, values cannot be equal.
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
 * @tparam T The first value entry type.
 * @tparam U The second value entry type.
 * @param a The first value to compare.
 * @param b The second value to compare.
 * @return Are the values different?
 */
template <size_t D, size_t E, typename T, typename U>
__host__ __device__ inline bool constexpr operator!=(const Cartesian<D, T>& a, const Cartesian<E, U>& b) noexcept
{
    return !(a == b);
}

#endif