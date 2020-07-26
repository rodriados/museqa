/**
 * Multiple Sequence Alignment functor header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2020 Rodrigo Siqueira
 */
#pragma once

#include <utils.hpp>

namespace msa
{
    /**#@+
     * Wraps a function pointer into a functor.
     * @tparam F The full function signature type.
     * @tparam R The function return type.
     * @tparam P The function parameter types.
     * @since 0.1.1
     */
    template <typename F>
    class functor
    {
        static_assert(std::is_function<F>::value, "a functor must have a function signature type");
    };

    template <typename R, typename ...P>
    class functor<R(P...)>
    {
        public:
            using return_type = R;                  /// The functor's return type.
            using function_type = R (*)(P...);      /// The functor's raw pointer type.

        protected:
            function_type m_function = nullptr;      /// The raw functor's pointer.

        public:
            __host__ __device__ inline constexpr functor() noexcept = default;
            __host__ __device__ inline constexpr functor(const functor&) noexcept = default;
            __host__ __device__ inline constexpr functor(functor&&) noexcept = default;

            /**
             * Instantiates a new functor.
             * @param function The function pointer to be encapsulated by functor.
             */
            __host__ __device__ inline constexpr functor(function_type function) noexcept
            :   m_function {function}
            {}

            __host__ __device__ inline functor& operator=(const functor&) noexcept = default;
            __host__ __device__ inline functor& operator=(functor&&) noexcept = default;

            /**
             * The functor call operator.
             * @tparam T The given parameter types.
             * @param param The given functor parameters.
             * @return The functor return value.
             */
            template <typename ...T>
            __host__ __device__ inline constexpr return_type operator()(T&&... param) const
            {
                return (m_function)(param...);
            }

            /**
             * Allows the raw functor type to be directly accessed or called.
             * @return The raw function pointer.
             */
            __host__ __device__ inline constexpr function_type operator&() const
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
    /**#@-*/
}
