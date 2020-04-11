/** 
 * Multiple Sequence Alignment cartesian header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#include <utility>

#include <utils.hpp>
#include <tuple.hpp>
#include <exception.hpp>
#include <reflection.hpp>

namespace msa
{
    namespace detail
    {
        namespace cartesian
        {
            /**
             * Represents the base of a D-dimensional cartesian value. This value
             * can then be expanded to or viewed as a point, a vector or ultimately
             * as a geometric space formed of the given size.
             * @tparam D The number of cartesian dimensions.
             * @tparam T The type of dimension values.
             * @since 0.1.1
             */
            template <size_t D, typename T>
            struct base : public reflector
            {
                static_assert(D > 0, "cartesian values are at least 1-dimensional");
                static_assert(std::is_integral<T>::value, "cartesian dimensions must be of integer type");

                using element_type = T;                     /// The cartesian dimension type.
                static constexpr size_t dimensionality = D; /// The cartesian dimensionality.

                element_type dim[D] = {};                   /// The cartesian dimensions values.

                __host__ __device__ inline constexpr base() noexcept = default;
                __host__ __device__ inline constexpr base(const base&) noexcept = default;
                __host__ __device__ inline constexpr base(base&&) noexcept = default;

                /**
                 * Constructs a new cartesian value instance.
                 * @tparam U The dimensional values' types.
                 * @param value The cartesian dimensional values.
                 */
                template <
                        class ...U
                    ,   class X = element_type
                    ,   class = typename std::enable_if<utils::all(std::is_convertible<U, X>::value...)>::type
                    >
                __host__ __device__ inline constexpr base(const U&... value) noexcept
                :   dim {static_cast<element_type>(value)...}
                {}

                __host__ __device__ inline base& operator=(const base&) noexcept = default;
                __host__ __device__ inline base& operator=(base&&) noexcept = default;

                /**
                 * Gives direct access to the value of one of the cartesian dimensions.
                 * @param offset The requested dimension offset identifier.
                 * @return The requested dimension's value.
                 */
                __host__ __device__ inline element_type operator[](ptrdiff_t offset) const
                {
                    enforce(size_t(offset) < D, "requested dimension is out of range");
                    return dim[offset];
                }

                /**
                 * The addition of two cartesian values. This is simply a mathematical
                 * addition of vectors.
                 * @param other The cartesian value to add to the current one.
                 * @return The new cartesian value.
                 */
                template <typename U>
                __host__ __device__ inline constexpr base operator+(const base<D, U>& other) const noexcept
                {
                    using namespace utils;
                    return from_tuple(zipwith(add<element_type>, tie(dim), tie(other.dim)));
                }

                /**
                 * The scalar product of a cartesian value.
                 * @tparam U The scalar's type.
                 * @param scalar The scalar to multiply the cartesian value by.
                 * @return The new cartesian value.
                 */
                template <typename U>
                __host__ __device__ inline constexpr base operator*(const U& scalar) const noexcept
                {
                    using namespace utils;
                    return from_tuple(apply(mul<element_type>, static_cast<element_type>(scalar), tie(dim)));
                }

                /**
                 * Collapses a multidimensional cartesian value into an 1-dimensional
                 * value, as if applying the given point in this space.
                 * @tparam U The other cartesian value's dimension type.
                 * @param other The cartesian value to collapse into the space.
                 * @return The 1-dimensional collapsed cartesian value.
                 */
                template <typename U = T>
                __host__ __device__ inline constexpr element_type collapse(const base<D, U>& other) const noexcept
                {
                    using namespace utils;
                    return foldl(add<element_type>, 0, zipwith(mul<element_type>, tie(other.dim), 
                        scanr(mul<element_type>, 1, tail(tie(dim)))));
                }

                /**
                 * Calculates the total cartesian space's volume.
                 * @return The cartesian space volume.
                 */
                __host__ __device__ inline constexpr element_type volume() const noexcept
                {
                    using namespace utils;
                    return foldl(mul<element_type>, element_type {1}, tie(dim));
                }

                /**
                 * Creates a new cartesian value instance from a tuple.
                 * @tparam I The sequence of tuple indeces.
                 * @tparam U The tuple types.
                 * @param value The tuple from which the new cartesian will be created.
                 * @return The new cartesian instance.
                 */
                template <size_t ...I, typename ...U>
                __host__ __device__ inline static constexpr base from_tuple(
                        const detail::tuple::base<indexer<I...>, U...>& value
                    ) noexcept
                {
                    return {detail::tuple::get<I>(value)...};
                }

                using reflex = decltype(reflect(dim));
            };
        }
    }

    /**
     * Represents a multidimensional cartesian value, that can be used either as
     * a space, a point or a vector representation.
     * @tparam D The cartesian dimensionality.
     * @tparam T The cartesina dimensions' type.
     * @since 0.1.1
     */
    template <size_t D, typename T = ptrdiff_t>
    struct cartesian : public detail::cartesian::base<D, T>
    {
        using underlying_type = detail::cartesian::base<D, T>;
        
        using underlying_type::base;
        using underlying_type::operator=;
    };

    template <typename T>
    struct cartesian<1, T> : public detail::cartesian::base<1, T>
    {
        using underlying_type = detail::cartesian::base<1, T>;
        
        using underlying_type::base;
        using underlying_type::operator=;

        /**
         * Allows a uni-dimensional cartesian value to be represented as a number.
         * @return The uni-dimensional cartesian value.
         */
        __host__ __device__ inline constexpr operator T() const noexcept
        {
            return this->dim[0];
        }
    };
    /**#@-*/
}
