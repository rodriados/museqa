/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A general functor abstraction implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <utility>

#include <museqa/utility.hpp>

namespace museqa
{
    namespace utility
    {
        /**
         * Wraps and manages a general function pointer.
         * @tparam F The given functor type.
         * @since 1.0
         */
        template <typename F>
        class functor
        {
            static_assert(std::is_function<F>::value, "a functor must have a function-like type");
        };

        /**
         * Wraps a general function by describing its return and parameters types.
         * @tparam R The functor's return type.
         * @tparam P The functor's parameter types.
         * @since 1.0
         */
        template <typename R, typename ...P>
        class functor<R(P...)>
        {
          public:
            using return_type = R;                  /// The functor's returning type.
            using function_type = R (*)(P...);      /// The functor's raw pointer type.

          protected:
            function_type m_function = nullptr;     /// The raw functor's pointer.

          public:
            __host__ __device__ inline constexpr functor() noexcept = default;
            __host__ __device__ inline constexpr functor(const functor&) noexcept = default;
            __host__ __device__ inline constexpr functor(functor&&) noexcept = default;

            /**
             * Instantiates a new functor.
             * @param function The function pointer to be encapsulated by the functor.
             */
            __host__ __device__ inline constexpr functor(function_type function) noexcept
              : m_function {function}
            {}

            __host__ __device__ inline functor& operator=(const functor&) noexcept = default;
            __host__ __device__ inline functor& operator=(functor&&) noexcept = default;

            /**
             * The functor call operator.
             * @tparam T The given parameter types.
             * @param param The given functor parameters.
             * @return The functor execution return value.
             */
            template <typename ...T>
            __host__ __device__ inline constexpr return_type operator()(T&&... params) const
            {
                return (m_function)(params...);
            }

            /**
             * Allows the raw functor type to be directly accessed or called.
             * @return The raw function pointer.
             */
            __host__ __device__ inline constexpr function_type operator*() const
            {
                return m_function;
            }

            /**
             * Checks whether the functor is empty or not.
             * @return Is the functor empty?
             */
            __host__ __device__ inline constexpr bool empty() const noexcept
            {
                return (m_function == nullptr);
            }
        };
    }

    namespace factory
    {
        /**
         * Builds a new functor from a generic function.
         * @tparam R The functor's return type.
         * @tparam P The functor's parameter types.
         * @param func The function to be wrapped in a new functor instance.
         */
        template <typename R, typename ...P>
        __host__ __device__ inline auto functor(R (*func)(P...)) noexcept -> utility::functor<R(P...)>
        {
            return {func};
        }
    }
}
