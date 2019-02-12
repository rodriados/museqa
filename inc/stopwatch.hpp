/** 
 * Multiple Sequence Alignment stopwatch header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef STOPWATCH_HPP_INCLUDED
#define STOPWATCH_HPP_INCLUDED

#include <ratio>
#include <chrono>
#include <utility>

#include "utils.hpp"

namespace stopwatch
{
    /**
     * The internal clock type used to calculate time intervals.
     * @since 0.1.1
     */
    using Clock = typename std::conditional<
            std::chrono::high_resolution_clock::is_steady
        ,   std::chrono::high_resolution_clock
        ,   std::chrono::steady_clock
        >::type;

    /**
     * Represents a time duration in milliseconds.
     * @since 0.1.1
     */
    using Duration = std::chrono::duration<double>;

    namespace duration
    {
        /**#@+
         * Types responsible for defining time durations. These types are directly
         * convertible to and from the original Duration type.
         * @see stopwatch::Duration
         * @since 0.1.1
         */
        using Milliseconds = std::chrono::duration<double, std::milli>;
        using Seconds      = std::chrono::duration<double, std::ratio<1, 1>>;
        using Minutes      = std::chrono::duration<double, std::ratio<60, 1>>;
        using Hours        = std::chrono::duration<double, std::ratio<3600, 1>>;
        using Days         = std::chrono::duration<double, std::ratio<86400, 1>>;
        /**#@-*/
    };

    /**
     * Represents a point in time.
     * @since 0.1.1
     */
    using TimePoint = std::chrono::time_point<Clock, Duration>;

    /**
     * Retrieves a time point representing the current point in time.
     * @return The current point in time.
     */
    inline TimePoint now()
    {
        return Clock::now();
    }

    /**
     * Retrieves the time elapsed since given time point.
     * @param point The time point used as duration start.
     * @return The time elapsed since given point.
     */
    inline Duration elapsed(const TimePoint& point)
    {
        return {now() - point};
    }

    /**
     * Times the execution of given function.
     * @tparam P The lambda function parameter types.
     * @param lambda The function to be executed.
     * @param params The function parameters.
     * @return The time spent by the function.
     */
    template <typename ...P>
    inline Duration run(Functor<void(P...)> lambda, P&&... params)
    {
        TimePoint start = now();
        std::forward<decltype(lambda)>(lambda) (std::forward<decltype(params)>(params)...);
        return elapsed(start);
    }
};

/**
 * This function allows duration instances to be directly printed into an ostream instance.
 * @tparam R A std::ratio type instance.
 * @param os The output stream object.
 * @param duration The duration to print.
 */
template <typename R>
std::ostream& operator<<(std::ostream& os, const std::chrono::duration<double, R>& duration)
{
    os << duration.count();
    return os;
}

#endif