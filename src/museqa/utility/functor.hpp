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
#include <museqa/utility/delegate.hpp>
#include <museqa/utility/tuple.hpp>

#include <museqa/memory/pointer/shared.hpp>
#include <museqa/memory/pointer/unmanaged.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace utility
{
    /**
     * A functor wrapper for bound or unbound functions. The wrapped function might
     * have any function type, including as a member function pointer. If the wrapped
     * function has member function type, an instance of the object to which the
     * function is bound to is necessary to create a functor.
     * @tparam F The wrapped function pointer type.
     * @tparam O The object type to which the function might be bound to.
     * @since 1.0
     */
    template <typename F, typename O = void>
    class functor;

    /**
     * Wraps a generic non-member function, callable from any context.
     * @tparam F The wrapped function pointer type.
     * @since 1.0
     */
    template <typename F>
    class functor<F, void> : public decltype(utility::delegate(std::declval<F>()))
    {
        private:
            typedef decltype(utility::delegate(std::declval<F>())) underlying_type;

        public:
            using underlying_type::delegate;

            /**
             * Unwraps and exposes the underlying function pointer.
             * @return The unwrapped function pointer.
             */
            __host__ __device__ inline constexpr F *unwrap() const
            {
                return this->m_function;
            }
    };

    namespace detail
    {
        /**
         * Analyses whether the given type is a member function pointer and if so,
         * isolates the member function's return type.
         * @tparam F The member function type to be analysed.
         * @since 1.0
         */
        template <typename F>
        struct analyzer : std::false_type {};

        /*
         * Defines auxiliary macros for creating all possible member function types.
         * Although quite repetitive, there's apparently no better way to do it.
         */
        #define __museqaanalyzer__(MF)                                         \
          template <typename R, typename T, typename ...P>                     \
          struct analyzer<MF> : std::true_type { typedef R result_type; };

        #define __museqarepeater1__(qq)                                        \
          __museqaanalyzer__(R(T::*)(P...) qq);                                \
          __museqaanalyzer__(R(T::*)(P......) qq);

        #define __museqarepeater2__(qq)                                        \
          __museqarepeater1__(qq);                                             \
          __museqarepeater1__(const qq);                                       \
          __museqarepeater1__(volatile qq);                                    \
          __museqarepeater1__(const volatile qq);

        /*
         * Creates every possible combination of qualifiers for a member function
         * type. Therefore, the return type of any member function can be now known.
         */
        __museqarepeater2__();
        __museqarepeater2__(&);
        __museqarepeater2__(&&);
        __museqarepeater2__(noexcept);
        __museqarepeater2__(& noexcept);
        __museqarepeater2__(&& noexcept);

        #undef __museqarepeater2__
        #undef __museqarepeater1__
        #undef __museqaanalyzer__
    }

    /**
     * Wraps a generic member function bound to an object instance.
     * @tparam F The wrapped member function pointer type.
     * @tparam O The object type to which the function is bound to.
     * @since 1.0
     */
    template <typename F, typename O>
    class functor
    {
        static_assert(std::is_class<O>::value, "functor must be bound to a class-like type");
        static_assert(detail::analyzer<F>::value, "only member functions can be bound to type");

        protected:
            typedef O object_type;
            typedef memory::pointer::shared<O> pointer_type;

        public:
            typedef F function_type;
            typedef typename detail::analyzer<F>::result_type result_type;

        private:
            mutable pointer_type m_object;
            function_type m_function = nullptr;

        public:
            __host__ __device__ inline constexpr functor() noexcept = default;
            __host__ __device__ inline functor(const functor&) noexcept = default;
            __host__ __device__ inline functor(functor&&) noexcept = default;

            /**
             * Builds a function from a non-owning pointer to the bound object.
             * @param object The object to which the member function is bound to.
             * @param lambda The function pointer to be wrapped by the functor.
             */
            __host__ __device__ inline explicit functor(object_type& object, function_type lambda) noexcept
              : functor {memory::pointer::unmanaged(&object), lambda}
            {}

            /**
             * Builds a functor by sharing ownership of the bound object.
             * @param object The pointer to the object the function is bound to.
             * @param lambda The function pointer to be wrapped by the functor.
             */
            __host__ __device__ inline functor(const pointer_type& object, function_type lambda) noexcept
              : m_object {object}
              , m_function {lambda}
            {}

            /**
             * Builds a functor by acquiring ownership of the bound object.
             * @param object The pointer to the object the function is bound to.
             * @param lambda The function pointer to be wrapped by the functor.
             */
            __host__ __device__ inline functor(pointer_type&& object, function_type lambda) noexcept
              : m_object {std::forward<decltype(object)>(object)}
              , m_function {lambda}
            {}

            __host__ __device__ inline functor& operator=(const functor&) __devicesafe__ = default;
            __host__ __device__ inline functor& operator=(functor&&) __devicesafe__ = default;

            /**
             * Invokes the wrapped function and returns the produced result.
             * @tparam P The given parameters' types.
             * @param param The parameters' to invoke the functor with.
             * @return The functor execution resulting value.
             */
            template <typename ...P>
            __host__ __device__ inline decltype(auto) operator()(P&&... param) const
            {
                museqa::guard(!empty(), "an empty functor cannot be invoked");
                return ((*m_object).*m_function)(std::forward<decltype(param)>(param)...);
            }

            /**
             * Unwraps and exposes the underlying function pointer and bound object.
             * @return The unwrapped function pointer and bound object tuple.
             */
            __host__ __device__ inline constexpr decltype(auto) unwrap() const
            {
                return tuple<object_type&, function_type> {*m_object, m_function};
            }

            /**
             * Checks whether the functor is callable or not.
             * @return Is the functor callable?
             */
            __host__ __device__ inline constexpr bool empty() const noexcept
            {
                return !m_object || (m_function == nullptr);
            }
    };
}

namespace factory
{
    /**
     * Creates a new functor from a plain unbound function.
     * @tparam F The given function type.
     * @param function The function to be wrapped.
     * @return The new functor instance.
     */
    template <typename F>
    __host__ __device__ inline constexpr auto functor(const F& function) noexcept
    -> typename std::enable_if<std::is_function<F>::value, museqa::utility::functor<F>>::type
    {
        return {function};
    }

  #if !MUSEQA_RUNTIME_DEVICE
    /**
     * Creates a new functor from a method bound to an object.
     * @tparam T The object type the method is bound to.
     * @tparam F The given method type.
     * @param object The object instance the functor is bound to.
     * @param method The method to be wrapped by functor.
     * @return The new functor instance.
     */
    template <typename T, typename F>
    __host__ inline auto functor(T&& object, const F& method) -> museqa::utility::functor<F, pure<T>>
    {
        return museqa::utility::functor<F, pure<T>>(
            museqa::memory::pointer::shared<pure<T>>(
                new pure<T>(std::forward<decltype(object)>(object))
              , [](void *ptr) { delete reinterpret_cast<pure<T>*>(ptr); })
          , method
        );
    }

  #else
    /**
     * Creates a new functor from a method bound to an object.
     * @tparam T The object type the method is bound to.
     * @tparam F The given method type.
     * @param object The object instance the functor is bound to.
     * @param method The method to be wrapped by functor.
     * @return The new functor instance.
     */
    template <typename T, typename F>
    __device__ inline auto functor(T&& object, const F& method) -> museqa::utility::functor<F, pure<T>>
    {
        return museqa::utility::functor<F, pure<T>>(
            museqa::memory::pointer::unmanaged<pure<T>>(&object)
          , method
        );
    }
  #endif

    /**
     * Creates a new functor from an invokable object, such as lambdas.
     * @tparam T The given invokable object type.
     * @param object The invokable object instance to be wrapped.
     * @return The new functor instance.
     */
    template <typename T>
    __host__ __device__ inline auto functor(T&& object)
    -> typename std::enable_if<
        std::is_class<pure<T>>::value
      , decltype(factory::functor(object, &pure<T>::operator()))
    >::type
    {
        return factory::functor(std::forward<decltype(object)>(object), &pure<T>::operator());
    }
}

MUSEQA_END_NAMESPACE
