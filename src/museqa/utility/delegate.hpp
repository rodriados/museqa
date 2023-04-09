/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A wrapper for a generic function pointer.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

#include <utility>

#include <museqa/environment.h>

#include <museqa/guard.hpp>
#include <museqa/utility.hpp>
#include <museqa/utility/tuple.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace utility
{
    /**
     * A delegate for a function pointer of generic return and parameter types.
     * The wrapped function may be variadic, therefore its last ellipses-parameter
     * is not representable by the delegate type. Nonetheless, a parameter type
     * to represent a function's variadic parameter is injected, at the end of the
     * parameter types list via deduction guides, allowing us to differentiate between
     * variadic and non-variadic versions of similar functions.
     * @tparam R The function's invocation return type.
     * @tparam P The function's parameter types.
     * @since 1.0
     */
    template <typename R, typename ...P>
    class delegate_t;

    namespace detail
    {
        /**
         * Auxiliary type for marking a variadic function delegate. As this type
         * is hidden from the library's public API, a function must never expect
         * a parameter to be of this type.
         * @since 1.0
         */
        struct variadic_t final
        {
            /**
             * Creates type of a non-variadic function with given parameter types.
             * @tparam R The function's return type.
             * @tparam P The function's typed-parameters type.
             * @return The non-variadic function type.
             */
            template <typename R, typename ...P>
            __host__ __device__ static constexpr auto create(tuple_t<P...>) -> identity_t<R(P...)>;

            /**
             * Creates type of a variadic function with given parameter types.
             * @tparam R The variadic function's return type.
             * @tparam P The variadic function's typed-parameters type.
             * @return The variadic function type.
             */
            template <typename R, typename ...P>
            __host__ __device__ static constexpr auto create(tuple_t<variadic_t, P...>) -> identity_t<R(P..., ...)>;
        };

        /**
         * A tuple representing the parameters of function. When dealing with variadic
         * functions, the variadic flag is brought to the beginning of the parameters
         * type list. Thus, a variadic function can be easily spotted by SFINAE.
         * @tparam P The function's parameters types.
         */
        template <typename ...P>
        using parameters_t = typename std::conditional<
                std::is_same<const variadic_t&, decltype(last(std::declval<tuple_t<int, P...>>()))>::value
              , decltype(init(std::declval<tuple_t<variadic_t, P...>>()))
              , tuple_t<P...>
            >::type;

        /**
         * The callable type of a generic delegate with potentially variadic type.
         * @tparam R The callable's return type.
         * @tparam P The callable's parameters' types and variadic flag.
         * @since 1.0
         */
        template <typename R, typename ...P>
        using callable_t = typename decltype(variadic_t::create<R>(std::declval<parameters_t<P...>>()))::type;
    }

    /**
     * A delegate for a function pointer of generic return and parameter types.
     * @tparam R The function's invocation return type.
     * @tparam P The function's parameter types.
     * @since 1.0
     */
    template <typename R, typename ...P>
    class delegate_t
    {
        public:
            using result_t = R;
            using function_t = detail::callable_t<R, P...>;

        protected:
            function_t *m_function = nullptr;

        public:
            __host__ __device__ inline constexpr delegate_t() noexcept = default;
            __host__ __device__ inline constexpr delegate_t(const delegate_t&) noexcept = default;
            __host__ __device__ inline constexpr delegate_t(delegate_t&&) noexcept = default;

            /**
             * Wraps a function pointer into a delegate instance.
             * @param function An invokable function pointer.
             */
            __host__ __device__ inline constexpr delegate_t(function_t *function) noexcept
              : m_function {function}
            {}

            __host__ __device__ inline constexpr delegate_t& operator=(const delegate_t&) noexcept = default;
            __host__ __device__ inline constexpr delegate_t& operator=(delegate_t&&) noexcept = default;

            /**
             * Invokes the wrapped function with the given parameters.
             * @tparam T The function invocation's given parameters.
             * @param params The given parameter values or references.
             * @return The wrapped function's invocation result.
             */
            template <typename ...T>
            __host__ __device__ inline constexpr decltype(auto) operator()(T&&... params) const
            {
                museqa::guard(!empty(), "an empty delegate cannot be invoked");
                return m_function (std::forward<decltype(params)>(params)...);
            }

            /**
             * Implicitly converts the delegate into its wrapped function pointer.
             * @return The wrapped function pointer.
             */
            __host__ __device__ inline constexpr operator function_t*() const
            {
                return m_function;
            }

            /**
             * Checks whether the delegate is empty or not.
             * @return Is the delegate empty?
             */
            __host__ __device__ inline constexpr bool empty() const noexcept
            {
                return (m_function == nullptr);
            }
    };

    /*
     * Deduction guides for generic function delegates.
     * @since 1.0
     */
    template <typename R, typename ...P> delegate_t(R(P...)) -> delegate_t<R, P...>;
    template <typename R, typename ...P> delegate_t(R(P..., ...)) -> delegate_t<R, P..., detail::variadic_t>;
}

MUSEQA_END_NAMESPACE
