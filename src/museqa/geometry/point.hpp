/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file An n-dimensional geometric point data structure.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <utility>

#include <museqa/environment.h>

#if defined(MUSEQA_COMPILER_NVCC)
  #pragma push
  #pragma diag_suppress = unrecognized_gcc_pragma
#endif

#include <fmt/format.h>

#if defined(MUSEQA_COMPILER_NVCC)
  #pragma pop
#endif

#include <museqa/utility.hpp>
#include <museqa/utility/tuple.hpp>
#include <museqa/utility/indexer.hpp>
#include <museqa/exception.hpp>

#if !defined(MUSEQA_AVOID_REFLECTION)
  #include <museqa/utility/reflection.hpp>
#endif


namespace museqa
{
    namespace geometry
    {
        /**
         * Represents a simple geometric D-dimensional point.
         * @tparam D The point's dimensionality.
         * @tparam T The point's dimensions' type.
         * @since 1.0
         */
        template <size_t D, typename T = int64_t>
        struct point;

        namespace impl
        {
            /**
             * The representation of the dimensions of a generic D-dimensional point.
             * As the number of dimensions is arbitrary, there are no special axis
             * identifiers to points of higher dimensions.
             * @tparam D The point's dimensionality.
             * @tparam T The point's dimensions' type.
             * @since 1.0
             */
            template <size_t D, typename T>
            struct point
            {
                typedef T dimension_type;

                static constexpr size_t dimensionality = D;
                static_assert(dimensionality > 0, "a point must be at least 1-dimensional");
                static_assert(std::is_arithmetic<dimension_type>(), "a point has arithmetic dimension type");

                union {
                    dimension_type dim[dimensionality] {};
                };
            };

            /**
             * The representation of a 1-dimensional point. As there is only one
             * single dimensional value to a point of such low-dimensionality, it
             * can be dealt with and interpreted as a scalar value.
             * @tparam T The point's dimension type.
             * @since 1.0
             */
            template <typename T>
            struct point<1, T>
            {
                typedef T dimension_type;

                static constexpr size_t dimensionality = 1;
                static_assert(std::is_arithmetic<dimension_type>(), "a point has arithmetic dimension type");

                union {
                    dimension_type value, dim[1] {};
                    struct { dimension_type a; };
                    struct { dimension_type i; };
                    struct { dimension_type x; };
                };

                /**
                 * Converts the point into its dimension value, enabling it for
                 * seamless usage as a scalar value due to its 1-dimension specialty.
                 * @return The point's 1-dimensional scalar value.
                 */
                __host__ __device__ inline operator dimension_type() const noexcept
                {
                    return value;
                }
            };

            /**
             * The representation of a 2-dimensional point. For this dimensionality,
             * a point may have its dimensions or axis named with or refered to
             * as different 2-letter combinations.
             * @tparam T The point's dimensions' type.
             * @since 1.0
             */
            template <typename T>
            struct point<2, T>
            {
                typedef T dimension_type;

                static constexpr size_t dimensionality = 2;
                static_assert(std::is_arithmetic<dimension_type>(), "a point has arithmetic dimension type");

                union {
                    dimension_type dim[2] {};
                    struct { dimension_type a, b; };
                    struct { dimension_type i, j; };
                    struct { dimension_type x, y; };
                };
            };

            /**
             * The representation of a 3-dimensional point. For this dimensionality,
             * a point may have its dimensions or axis named with or refered to
             * as different 3-letter combinations.
             * @tparam T The point's dimensions' type.
             * @since 1.0
             */
            template <typename T>
            struct point<3, T>
            {
                typedef T dimension_type;

                static constexpr size_t dimensionality = 3;
                static_assert(std::is_arithmetic<dimension_type>(), "a point has arithmetic dimension type");

                union {
                    dimension_type dim[3] {};
                    struct { dimension_type a, b, c; };
                    struct { dimension_type i, j, k; };
                    struct { dimension_type x, y, z; };
                };
            };

            /**
             * The representation of a 4-dimensional point. This is the highest
             * dimensionality for which a point may have its dimensions or axis
             * named with or refered to as different letters combinations.
             * @tparam T The point's dimensions' type.
             * @since 1.0
             */
            template <typename T>
            struct point<4, T>
            {
                typedef T dimension_type;

                static constexpr size_t dimensionality = 4;
                static_assert(std::is_arithmetic<dimension_type>(), "a point has arithmetic dimension type");

                union {
                    dimension_type dim[4] {};
                    struct { dimension_type a, b, c, d; };
                    struct { dimension_type x, y, z, w; };
                };
            };
        }

        /**
         * Represents a simple generic geometric D-dimensional point.
         * @tparam D The point's dimensionality.
         * @tparam T The point's dimensions' type.
         * @since 1.0
         */
        template <size_t D, typename T>
        class point : public geometry::impl::point<D, T>
        {
          private:
            typedef geometry::impl::point<D, T> underlying_type;

          public:
            using typename underlying_type::dimension_type;
            using underlying_type::dimensionality;

          public:
            __host__ __device__ inline constexpr point() noexcept = default;
            __host__ __device__ inline constexpr point(const point&) noexcept = default;
            __host__ __device__ inline constexpr point(point&&) noexcept = default;

            /**
             * Instantiates a new point from a general list of dimension values.
             * @tparam U The list of dimensions parameter types.
             * @param dim The point's dimensions' values.
             */
            template <
                class ...U
              , class = typename std::enable_if<utility::all(std::is_convertible<U, dimension_type>{}...)>::type
              , class = typename std::enable_if<dimensionality == sizeof...(U)>::type
            >
            __host__ __device__ inline constexpr point(const U&... dim)
              : underlying_type {static_cast<dimension_type>(dim)...}
            {}

            /**
             * Instantiates a new point from a tuple.
             * @tparam I The sequence tuple's indeces.
             * @tparam U The tuple's contents types.
             * @param tuple The tuple to build a point from.
             */
            template <size_t ...I, typename ...U>
            __host__ __device__ inline constexpr point(const utility::tuple<utility::indexer<I...>, U...>& tuple)
              : point {tuple.template get<I>()...}
            {}

            /**
             * Instantiates a new point from a foreign point instance.
             * @tparam U The foreign point's dimensions' type.
             * @param other The foreign point instance.
             */
            template <typename U>
            __host__ __device__ inline constexpr point(const point<dimensionality, U>& other)
              : point {typename utility::indexer<dimensionality>::type (), other}
            {}

            __host__ __device__ inline point& operator=(const point&) noexcept = default;
            __host__ __device__ inline point& operator=(point&&) noexcept = default;

            /**
             * Gives direct access to a point's dimension value.
             * @param offset The request point dimension offset.
             * @return The point's requested dimension value.
             */
            __host__ __device__ inline dimension_type& operator[](ptrdiff_t offset) noexcept(!safe)
            {
                museqa::ensure((size_t) offset < dimensionality, "point dimension out of range");
                return this->dim[offset];
            }

            /**
             * Gives direct access to a const-qualified point's dimension value.
             * @param offset The request point dimension offset.
             * @return The point's requested const-qualified dimension value.
             */
            __host__ __device__ inline const dimension_type& operator[](ptrdiff_t offset) const noexcept(!safe)
            {
                museqa::ensure((size_t) offset < dimensionality, "point dimension out of range");
                return this->dim[offset];
            }

          private:
            /**
             * Instantiates a new point from a foreign point and an indexer helper.
             * @tparam I The foreign point's dimensions sequence indeces.
             * @tparam P The foreign point's instance type.
             * @param other The foreign point to build a new point from.
             */
            template <size_t ...I, typename P>
            __host__ __device__ inline constexpr point(utility::indexer<I...>, const P& other)
              : point {other.dim[I]...}
            {}
        };

        /**
         * The equality operator for two points of equal dimensionality.
         * @tparam D The points' dimensionality value.
         * @tparam T The first point's dimension type.
         * @tparam U The second point's dimension type.
         * @param a The first point instance.
         * @param b The second point instance.
         * @return Are both points equal?
         */
        template <size_t D, typename T, typename U>
        __host__ __device__ inline constexpr bool operator==(const point<D, T>& a, const point<D, U>& b) noexcept
        {
            return utility::foldl(
                utility::andl<bool>, true
              , utility::zipwith(utility::eq<T, U>, utility::tie(a.dim), utility::tie(b.dim))
            );
        }

        /**
         * The equality operator for points with different dimensionality.
         * @tparam A The first point's dimensionality value.
         * @tparam B The second point's dimensionality value.
         * @tparam T The first point's dimension type.
         * @tparam U The second point's dimension type.
         * @return Are both points equal?
         */
        template <size_t A, size_t B, typename T, typename U>
        __host__ __device__ inline constexpr bool operator==(const point<A, T>&, const point<B, U>&) noexcept
        {
            return false;
        }

        /**
         * The inequality operator for two generic points.
         * @tparam A The first point's dimensionality value.
         * @tparam B The second point's dimensionality value.
         * @tparam T The first point's dimension type.
         * @tparam U The second point's dimension type.
         * @param a The first point instance.
         * @param b The second point instance.
         * @return Are the points different?
         */
        template <size_t A, size_t B, typename T, typename U>
        __host__ __device__ inline constexpr bool operator!=(const point<A, T>& a, const point<B, U>& b) noexcept
        {
            return !operator==(a, b);
        }

        /**
         * The distance operator for two generic points.
         * @tparam D The points' dimensionality value.
         * @tparam T The first point's dimension type.
         * @tparam U The second point's dimension type.
         * @param a The first point instance.
         * @param b The second point instance.
         * @return The Euclidean distance between the points.
         */
        template <size_t D, typename T, typename U>
        __host__ __device__ inline constexpr double distance(const point<D, T>& a, const point<D, U>& b) noexcept
        {
            return sqrt(utility::foldl(
                utility::add<double>, 0.0
              , utility::zipwith(
                    [](const T& a, const U& b) { return pow(b - a, 2.0); }
                  , utility::tie(a.dim)
                  , utility::tie(b.dim)
                )
            ));
        }
    }

  #if !defined(MUSEQA_AVOID_REFLECTION)
    /**
     * Explicitly defines the reflector for a point. Although a trivial type, a
     * point cannot be automatically reflected over due to its customized constructors.
     * @tparam D The point's dimensionality.
     * @tparam T The point's dimensions' type.
     * @since 1.0
     */
    template <size_t D, typename T>
    class utility::reflector<geometry::point<D, T>>
      : public utility::reflector<decltype(geometry::point<D, T>::dim)>
    {};
  #endif
}

/**
 * Implements a string formatter for a generic point type, thus allowing points
 * to be seamlessly printed as scalar types.
 * @tparam D The point's dimensionality.
 * @tparam T The point's dimensions' type.
 * @since 1.0
 */
template <size_t D, typename T>
class fmt::formatter<museqa::geometry::point<D, T>>
{
  private:
    typedef museqa::geometry::point<D, T> target_type;
    static constexpr size_t count = target_type::dimensionality;

  public:
    /**
     * Evaluates the formatter's parsing context.
     * @tparam C The parsing context type.
     * @param ctx The parsing context instance.
     * @return The processed and evaluated parsing context.
     */
    template <typename C>
    auto parse(C& ctx) -> decltype(ctx.begin())
    {
        return ctx.begin();
    }

    /**
     * Formats the point into a printable string.
     * @tparam F The formatting context type.
     * @param point The point to be formatted into a string.
     * @param ctx The formatting context instance.
     * @return The formatting context instance.
     */
    template <typename F>
    auto format(const target_type& point, F& ctx) -> decltype(ctx.out())
    {
        return fmt::format_to(ctx.out(), "({})", fmt::join(point.dim, point.dim + count, ", "));
    }
};
