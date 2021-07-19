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
         * Wraps a general member function pointer bound to an object type.
         * @tparam F The given member function pointer type.
         * @tparam O The type to which the functor is bound to.
         * @since 1.0
         */
        template <typename F, typename O = void>
        class functor
        {
            static_assert(std::is_class<O>::value, "a method functor must be bound to a class type");
            static_assert(std::is_member_function_pointer<F>::value, "only member functors can be bound to a type");

          private:
            struct analyzer;

          protected:
            using bound_type = O;                   /// The type the functor is bound to.
            using function_type = F;                /// The functor's raw pointer type.

          public:
            using result_type = decltype(analyzer::get(std::declval<F>()));

          protected:
            mutable bound_type m_object;            /// A pointer to the object the functor is bound to.
            function_type m_function = nullptr;     /// The raw functor's pointer.

          public:
            __host__ __device__ inline constexpr functor() noexcept = default;
            __host__ __device__ inline constexpr functor(const functor&) noexcept = default;
            __host__ __device__ inline constexpr functor(functor&&) noexcept = default;

            /**
             * Instantiates a new functor.
             * @param object The object to which the functor is bound to.
             * @param function The function pointer to be encapsulated by the functor.
             */
            __host__ __device__ inline constexpr functor(const bound_type& object, function_type function) noexcept
              : m_object {object}
              , m_function {function}
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
            __host__ __device__ inline result_type operator()(T&&... params) const
            {
                return (m_object.*m_function)(std::forward<decltype(params)>(params)...);
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

          private:
            /**
             * Auxiliary structure for extracting a function pointer's result type.
             * @since 1.0
             */
            struct analyzer
            {
                template <typename R, typename ...P> static R get(R (O::*)(P...));
                template <typename R, typename ...P> static R get(R (O::*)(P...) &);
                template <typename R, typename ...P> static R get(R (O::*)(P...) &&);
                template <typename R, typename ...P> static R get(R (O::*)(P......));
                template <typename R, typename ...P> static R get(R (O::*)(P......) &);
                template <typename R, typename ...P> static R get(R (O::*)(P......) &&);
                template <typename R, typename ...P> static R get(R (O::*)(P...) const);
                template <typename R, typename ...P> static R get(R (O::*)(P...) const &);
                template <typename R, typename ...P> static R get(R (O::*)(P...) const &&);
                template <typename R, typename ...P> static R get(R (O::*)(P......) const);
                template <typename R, typename ...P> static R get(R (O::*)(P......) const &);
                template <typename R, typename ...P> static R get(R (O::*)(P......) const &&);
                template <typename R, typename ...P> static R get(R (O::*)(P...) volatile);
                template <typename R, typename ...P> static R get(R (O::*)(P...) volatile &);
                template <typename R, typename ...P> static R get(R (O::*)(P...) volatile &&);
                template <typename R, typename ...P> static R get(R (O::*)(P......) volatile);
                template <typename R, typename ...P> static R get(R (O::*)(P......) volatile &);
                template <typename R, typename ...P> static R get(R (O::*)(P......) volatile &&);
                template <typename R, typename ...P> static R get(R (O::*)(P...) const volatile);
                template <typename R, typename ...P> static R get(R (O::*)(P...) const volatile &);
                template <typename R, typename ...P> static R get(R (O::*)(P...) const volatile &&);
                template <typename R, typename ...P> static R get(R (O::*)(P......) const volatile);
                template <typename R, typename ...P> static R get(R (O::*)(P......) const volatile &);
                template <typename R, typename ...P> static R get(R (O::*)(P......) const volatile &&);
            };
        };

        /**
         * Wraps a general function pointer.
         * @tparam F The given function pointer type.
         * @since 1.0
         */
        template <typename F>
        class functor<F, void>
        {
            static_assert(std::is_function<F>::value, "a functor must have a function-like type");

          private:
            struct analyzer;

          protected:
            using function_type = F*;               /// The functor's raw pointer type.

          public:
            using result_type = decltype(analyzer::get(std::declval<F*>()));

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
            __host__ __device__ inline constexpr result_type operator()(T&&... params) const
            {
                return (m_function)(std::forward<decltype(params)>(params)...);
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

          private:
            /**
             * Auxiliary structure for extracting a function pointer's result type.
             * @since 1.0
             */
            struct analyzer
            {
                template <typename R, typename ...P> static R get(R (*)(P...));
                template <typename R, typename ...P> static R get(R (*)(P......));
            };
        };
    }

    namespace factory
    {
        /**
         * Creates a new functor from a plain and simple function.
         * @tparam F The given function type.
         * @return The new functor instance.
         */
        template <typename F>
        __host__ __device__ inline constexpr auto functor(const F& function) noexcept
        -> typename std::enable_if<std::is_function<F>::value, utility::functor<F>>::type
        {
            return {function};
        }

        /**
         * Creates a new functor from a method bound to an object type.
         * @tparam O The object type the method is bound to.
         * @tparam F The given method type.
         * @return The new functor instance.
         */
        template <typename O, typename F>
        __host__ __device__ inline constexpr auto functor(const O& object, const F& method) noexcept
        -> typename std::enable_if<std::is_member_function_pointer<F>::value, utility::functor<F, O>>::type
        {
            return {object, method};
        }

        /**
         * Creates a new functor from an invokable object, such as lambdas.
         * @tparam T The given invokable object type.
         * @return The new functor instance.
         */
        template <typename T>
        __host__ __device__ inline constexpr auto functor(const T& object) noexcept
        -> typename std::enable_if<std::is_object<T>::value, decltype(functor(object, &T::operator()))>::type
        {
            return functor(object, &T::operator());
        }
    }
}
