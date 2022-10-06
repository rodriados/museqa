/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A generic n-dimensional coordinate system data structure.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>
#include <utility>

#include <museqa/environment.h>
#include <museqa/utility.hpp>
#include <museqa/utility/tuple.hpp>
#include <museqa/utility/reflection.hpp>

#include <museqa/thirdparty/fmtlib.h>

MUSEQA_BEGIN_NAMESPACE

namespace geometry
{
    /**
     * The representation of a coordinate system for a generic D-dimensional space.
     * Due to the purposedly arbitrary number of dimensions, there are no special
     * axis or dimension identifiers to coordinates higher than 4 dimensions.
     * @tparam D The coordinate system's space dimensionality.
     * @tparam T The type for the coordinate dimensions' values.
     * @since 1.0
     */
    template <size_t D, typename T = int64_t>
    struct coordinate
    {
        typedef T coordinate_type;
        inline static constexpr size_t dimensionality = D;

        static_assert(dimensionality > 0, "a coordinate must be at least 1-dimensional");
        static_assert(std::is_arithmetic<coordinate_type>(), "a coordinate must have arithmetic type");

        union {
            coordinate_type value[D] {};
        };
    };

    /**
     * The representation of a coordinate system for a generic 1-dimensional space.
     * As an 1-dimensional value is a scalar, such coordinate system can be interpreted
     * as and seamlessly converted to its scalar type.
     * @tparam T The type for the coordinate dimension's value.
     * @since 1.0
     */
    template <typename T>
    struct coordinate<1, T>
    {
        typedef T coordinate_type;
        inline static constexpr size_t dimensionality = 1;

        static_assert(std::is_arithmetic<coordinate_type>(), "a coordinate must have arithmetic type");

        union {
            coordinate_type value[1] {};
            struct { coordinate_type a; };
            struct { coordinate_type i; };
            struct { coordinate_type x; };
        };

        /**
         * Converts the coordinate value into its scalar value, a particularity
         * possible only 1-dimensional coordinate systems.
         * @return The point's 1-dimensional scalar value.
         */
        __host__ __device__ inline operator coordinate_type() const noexcept
        {
            return value;
        }
    };

    /**
     * The representation of a coordinate system for a generic 2-dimensional space.
     * At this dimensionality, the coordinates in the system might be referred to
     * as or named with different 2-letter combinations.
     * @tparam T The type for the coordinate dimension's value.
     * @since 1.0
     */
    template <typename T>
    struct coordinate<2, T>
    {
        typedef T coordinate_type;
        inline static constexpr size_t dimensionality = 2;

        static_assert(std::is_arithmetic<coordinate_type>(), "a coordinate must have arithmetic type");

        union {
            coordinate_type value[2];
            struct { coordinate_type a, b; };
            struct { coordinate_type i, j; };
            struct { coordinate_type x, y; };
        };
    };

    /**
     * The representation of a coordinate system for a generic 3-dimensional space.
     * At this dimensionality, the coordinates in the system might be referred to
     * as or named with different 3-letter combinations.
     * @tparam T The type for the coordinate dimension's value.
     * @since 1.0
     */
    template <typename T>
    struct coordinate<3, T>
    {
        typedef T coordinate_type;
        inline static constexpr size_t dimensionality = 3;

        static_assert(std::is_arithmetic<coordinate_type>(), "a coordinate must have arithmetic type");

        union {
            coordinate_type value[3];
            struct { coordinate_type a, b, c; };
            struct { coordinate_type i, j, k; };
            struct { coordinate_type x, y, z; };
        };
    };

    /*
     * Deduction guides for generic coordinate types.
     * @since 1.0
     */
    template <typename T, typename ...U> coordinate(T, U...)
        -> coordinate<sizeof...(U) + 1, T>;

    /**
     * The equality operator for two coordinates of equal dimensionalities.
     * @tparam D The coordinates' dimensionality value.
     * @tparam T The first coordinate's dimension type.
     * @tparam U The second coordinate's dimension type.
     * @param a The first coordinate instance.
     * @param b The second coordinate instance.
     * @return Are both coordinates equal?
     */
    template <size_t D, typename T, typename U>
    __host__ __device__ inline constexpr bool operator==(
        const coordinate<D, T>& a, const coordinate<D, U>& b
    ) noexcept {
        return utility::foldl(
            utility::andl, true
          , utility::zipwith(utility::equ, utility::tie(a.value), utility::tie(b.value))
        );
    }

    /**
     * The equality operator for coodinates with different dimensionalities.
     * @tparam A The first coordinate's dimensionality value.
     * @tparam B The second coordinate's dimensionality value.
     * @tparam T The first coordinate's dimension type.
     * @tparam U The second coordinate's dimension type.
     * @return Are both coodinates equal?
     */
    template <size_t A, size_t B, typename T, typename U>
    __host__ __device__ inline constexpr bool operator==(
        const coordinate<A, T>&, const coordinate<B, U>&
    ) noexcept {
        return false;
    }

    /**
     * The inequality operator for two generic coordinates.
     * @tparam A The first coordinate's dimensionality value.
     * @tparam B The second coordinate's dimensionality value.
     * @tparam T The first coordinate's dimension type.
     * @tparam U The second coordinate's dimension type.
     * @param a The first coordinate instance.
     * @param b The second coordinate instance.
     * @return Are the coordinates different?
     */
    template <size_t A, size_t B, typename T, typename U>
    __host__ __device__ inline constexpr bool operator!=(
        const coordinate<A, T>& a, const coordinate<B, U>& b
    ) noexcept {
        return !geometry::operator==(a, b);
    }
}

#if !defined(MUSEQA_AVOID_REFLECTION)

/**
 * Explicitly defines the reflector for a coordinate. Although a trivial type, a
 * coordinate cannot be automatically reflected over due to its internal union.
 * @tparam D The coordinate's dimensionality.
 * @tparam T The coordinate's dimensions' type.
 * @since 1.0
 */
template <size_t D, typename T>
class utility::reflector<geometry::coordinate<D, T>>
  : public utility::reflector<decltype(geometry::coordinate<D, T>::value)> {};

#endif

MUSEQA_END_NAMESPACE

#if !defined(MUSEQA_AVOID_FMTLIB)

/**
 * Implements a string formatter for a generic coordinate type, therefore allowing
 * coordinates to be seamlessly printed as well as scalar types.
 * @tparam D The coordinate's dimensionality.
 * @tparam T The coordinate's dimensions' type.
 * @since 1.0
 */
template <size_t D, typename T>
struct fmt::formatter<museqa::geometry::coordinate<D, T>>
{
    typedef museqa::geometry::coordinate<D, T> target_type;
    inline static constexpr size_t count = target_type::dimensionality;

    /**
     * Evaluates the formatter's parsing context.
     * @tparam C The parsing context type.
     * @param ctx The parsing context instance.
     * @return The processed and evaluated parsing context.
     */
    template <typename C>
    constexpr auto parse(C& ctx) const -> decltype(ctx.begin())
    {
        return ctx.begin();
    }

    /**
     * Formats the coordinate into a printable string.
     * @tparam F The formatting context type.
     * @param coordinate The coordinate to be formatted into a string.
     * @param ctx The formatting context instance.
     * @return The formatting context instance.
     */
    template <typename F>
    auto format(const target_type& coordinate, F& ctx) const -> decltype(ctx.out())
    {
        return fmt::format_to(
            ctx.out(), "({})"
          , fmt::join(coordinate.value, coordinate.value + count, ", ")
        );
    }
};

#endif
