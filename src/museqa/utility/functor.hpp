/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A general functor abstraction implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <utility>

#include <museqa/environment.h>
#include <museqa/guard.hpp>
#include <museqa/utility.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace utility
{
    /**
     * A functor wrapper for bounded or unbounded functions. The wrapped function
     * might have any function type, including a member function pointer. If the
     * wrapped function is of a member function type, an instance of the object
     * to which the function is bounded must be its first parameter when called.
     * @tparam F The wrapped function pointer type.
     * @tparam O The object type to which the function might be bounded to.
     * @since 1.0
     */
    template <typename F, typename O = void>
    class functor_t;

    namespace detail
    {
        /**
         * Extracts the return type of a generic function type.
         * @tparam F The function type to be analyzed.
         * @since 1.0
         */
        template <typename F>
        struct function_return_t;

        /*
         * Auxiliary macros for creating every possible function type with qualifiers.
         * Although quite repetitive, there's apparently no better way to do it.
         */
        #define MUSEQA_DECLARE_FN_RESULT_T(F)                   \
          template <typename R, typename ...P>                  \
          struct function_return_t<F> : identity_t<R> {};

        #define MUSEQA_DECLARE_FN_RESULT_T_VARIADIC(Q)          \
          MUSEQA_DECLARE_FN_RESULT_T(R(P...) Q)                 \
          MUSEQA_DECLARE_FN_RESULT_T(R(P..., ...) Q)

        #define MUSEQA_DECLARE_FN_RESULT_T_QUALIFIER(E)         \
          MUSEQA_DECLARE_FN_RESULT_T_VARIADIC(E)                \
          MUSEQA_DECLARE_FN_RESULT_T_VARIADIC(const E)          \
          MUSEQA_DECLARE_FN_RESULT_T_VARIADIC(volatile E)       \
          MUSEQA_DECLARE_FN_RESULT_T_VARIADIC(const volatile E)

        /*
         * Creates every possible combination of qualifiers for generic function types.
         * Therefore, the return type of any function can be now known.
         */
        MUSEQA_DECLARE_FN_RESULT_T_QUALIFIER()
        MUSEQA_DECLARE_FN_RESULT_T_QUALIFIER(&)
        MUSEQA_DECLARE_FN_RESULT_T_QUALIFIER(&&)
        MUSEQA_DECLARE_FN_RESULT_T_QUALIFIER(noexcept)
        MUSEQA_DECLARE_FN_RESULT_T_QUALIFIER(& noexcept)
        MUSEQA_DECLARE_FN_RESULT_T_QUALIFIER(&& noexcept)

        #undef MUSEQA_DECLARE_FN_RESULT_T_QUALIFIER
        #undef MUSEQA_DECLARE_FN_RESULT_T_VARIADIC
        #undef MUSEQA_DECLARE_FN_RESULT_T

        /**
         * Creates the type for a pointer to a bounded function.
         * @tparam F The function type to get a bounded pointer to.
         * @tparam O The object type to bind the function to.
         * @return The bounded function pointer type.
         */
        template <
            typename F
          , typename O
          , typename = std::enable_if_t<std::is_function_v<F> && !std::is_void_v<O>>>
        MUSEQA_CUDA_CONSTEXPR auto make_function_pointer() -> F O::*;

        /**
         * Creates the type for a pointer to an unbounded function.
         * @tparam F The function type to get a pointer to.
         * @tparam O Should be void for unbounded functions.
         * @return The unbounded function pointer type.
         */
        template <
            typename F
          , typename O
          , typename = std::enable_if_t<std::is_function_v<F> && std::is_void_v<O>>>
        MUSEQA_CUDA_CONSTEXPR auto make_function_pointer() -> F*;
    }

    /**
     * A function wrapper for bounded or unbouded functions.
     * @tparam F The wrapped function pointer type.
     * @tparam O The object type to which the function might be bounded to.
     * @since 1.0
     */
    template <typename F, typename O>
    class functor_t
    {
        static_assert(std::is_function_v<F>, "only functions are functors");
        static_assert(std::is_class_v<O> || std::is_void_v<O>
          , "a functor must be either unbounded or bounded to an object type");

        public:
            using function_t = F;
            using bound_object_t = O;
            using result_t = typename detail::function_return_t<F>::type;

        private:
            using function_pointer_t = decltype(detail::make_function_pointer<F, O>());

        private:
            function_pointer_t m_function = nullptr;

        public:
            MUSEQA_CONSTEXPR functor_t() noexcept = default;
            MUSEQA_CONSTEXPR functor_t(const functor_t&) noexcept = default;
            MUSEQA_CONSTEXPR functor_t(functor_t&&) noexcept = default;

            /**
             * Initializes a functor from a raw function or member function pointer.
             * @param function An invokable function pointer.
             */
            MUSEQA_CUDA_CONSTEXPR functor_t(function_pointer_t function) noexcept
              : m_function (function)
            {}

            MUSEQA_INLINE functor_t& operator=(const functor_t&) noexcept = default;
            MUSEQA_INLINE functor_t& operator=(functor_t&&) noexcept = default;

            /**
             * Invokes the wrapped functor and returns the produced result.
             * @tparam P The invocation parameters' types.
             * @param params The parameters' to invoke the functor with.
             * @return The functor execution resulting value.
             */
            template <typename ...P>
            MUSEQA_CUDA_CONSTEXPR decltype(auto) operator()(P&&... params) const
            {
                guard(!empty(), "cannot invoke empty functor");
                return utility::invoke(m_function, std::forward<decltype(params)>(params)...);
            }

            /**
             * Implicitly unwraps the underlying function pointer.
             * @return The unwrapped function pointer.
             */
            MUSEQA_CUDA_CONSTEXPR operator function_pointer_t() const
            {
                return unwrap();
            }

            /**
             * Unwraps the underlying function pointer.
             * @return The unwrapped function pointer.
             */
            MUSEQA_CUDA_CONSTEXPR function_pointer_t unwrap() const noexcept
            {
                return m_function;
            }

            /**
             * Checks whether the functor is callable or not.
             * @return Is the functor callable?
             */
            MUSEQA_CUDA_CONSTEXPR bool empty() const noexcept
            {
                return m_function == nullptr;
            }
    };

    /*
     * Deduction guides for generic functors. These guides are responsible for correctly
     * identifying whether the function is bounded or not.
     * @since 1.0
     */
    template <typename R, typename ...P> functor_t(R(*)(P...)) -> functor_t<R(P...)>;
    template <typename R, typename ...P> functor_t(R(*)(P..., ...)) -> functor_t<R(P..., ...)>;
    template <typename F, typename O> functor_t(F O::*) -> functor_t<F, O>;
}

MUSEQA_END_NAMESPACE
