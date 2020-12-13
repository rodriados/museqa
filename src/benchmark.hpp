/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements a function timing benchmark interface.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <ratio>
#include <chrono>
#include <utility>

namespace museqa
{
    namespace benchmark
    {
        /**
         * Represents a time duration in a given time frame.
         * @tparam T The type to represent duration with.
         * @tparam R The time ratio in relation to seconds.
         * @since 0.1.1
         */
        template <typename T, typename R = std::ratio<1>>
        struct duration : public std::chrono::duration<T, R>
        {
            using ratio = R;            /// The time ratio in relation to seconds.
            using value_type = T;       /// The scalar type to represent duration.
            
            using std::chrono::duration<T, R>::duration;

            /**
             * Converts a duration to an printable type.
             * @return The converted duration.
             */
            inline operator value_type() const noexcept
            {
                return this->count();
            }
        };

        /**
         * The internal clock type used for timing and calculating time intervals.
         * @since 0.1.1
         */
        using ticker = typename std::conditional<
                std::chrono::high_resolution_clock::is_steady
            ,   std::chrono::high_resolution_clock
            ,   std::chrono::steady_clock
            >::type;

        /**
         * Represents a point in time.
         * @tparam T The scalar type to which time point must be represented by.
         * @since 0.1.1
         */
        template <typename T>
        using time_point = std::chrono::time_point<ticker, duration<T>>;

        /**
         * Retrieves the current real-life time point.
         * @tparam T The scalar type to which time point must be represented by.
         * @return The current point in time.
         */
        template <typename T = double>
        inline auto now() noexcept -> time_point<T>
        {
            return ticker::now();
        }

        /**
         * Retrieves the time elapsed since given time point.
         * @tparam T The scalar type to which duration must be represented by.
         * @param point The initial duration start point.
         * @return The time elapsed since given point.
         */
        template <typename T>
        inline auto elapsed(const time_point<T>& point) noexcept -> duration<T>
        {
            return ticker::now() - point;
        }

        /**#@+
         * Benchmarks the execution of a given functor.
         * @tparam R The functor's return type.
         * @tparam T The scalar type to which duration must be represented by.
         * @tparam F The given functor type.
         * @tparam P The functor's parameter types.
         * @param ret The functor's execution return value.
         * @param lambda The functor to be executed.
         * @param params The functor's parameters.
         * @return The time spent by functor's execution.
         */
        template <typename T = double, typename F, typename ...P>
        inline auto run(F&& lambda, P&&... params) -> duration<T>
        {
            const time_point<T> start = now<T>();
            lambda(std::forward<decltype(params)>(params)...);
            return elapsed(start);
        }

        template <typename R, typename T = double, typename F, typename ...P>
        inline auto run(R& ret, F&& lambda, P&&... params)
        -> typename std::enable_if<std::is_copy_assignable<R>::value, duration<T>>::type
        {
            const time_point<T> start = now<T>();
            ret = lambda(std::forward<decltype(params)>(params)...);
            return elapsed(start);
        }
        /**#@-*/
    }
}
