/** 
 * Multiple Sequence Alignment benchmarking header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef BENCHMARK_HPP_INCLUDED
#define BENCHMARK_HPP_INCLUDED

#include <ratio>
#include <chrono>
#include <utility>

#include "utils.hpp"

namespace benchmark
{
    /**
     * The internal clock type used for timing and calculating time intervals.
     * @since 0.1.1
     */
    using Timekeeper = typename std::conditional<
            std::chrono::high_resolution_clock::is_steady
        ,   std::chrono::high_resolution_clock
        ,   std::chrono::steady_clock
        >::type;

    /**
     * Represents a time duration in a given time frame.
     * @since 0.1.1
     */
    template <typename T, typename R = std::ratio<1>>
    struct Duration : public std::chrono::duration<T, R>
    {
        using std::chrono::duration<T, R>::duration;

        /**
         * Converts a duration to an printable type.
         * @return The converted duration.
         */
        inline operator T() const noexcept
        {
            return this->count();
        }
    };

    /**
     * Represents a point in time.
     * @since 0.1.1
     */
    using TimePoint = std::chrono::time_point<Timekeeper, Duration<double>>;

    /**#@+
     * Types responsible for defining different time durations. These are
     * directly convertible to and from the original Duration type.
     * @see benchmark::Duration
     * @since 0.1.1
     */
    using Milliseconds  = Duration<double, std::milli>;
    using Seconds       = Duration<double, std::ratio<1>>;
    using Minutes       = Duration<double, std::ratio<60>>;
    using Hours         = Duration<double, std::ratio<3600>>;
    /**#@-*/

    /**
     * Retrieves a time point representing the current point in time.
     * @return The current point in time.
     */
    inline TimePoint now() noexcept
    {
        return Timekeeper::now();
    }

    /**
     * Retrieves the time elapsed since given time point.
     * @param point The initial duration start point.
     * @return The time elapsed since given point.
     */
    inline Duration<double> elapsed(const TimePoint& point) noexcept
    {
        return {Timekeeper::now() - point};
    }

    /**
     * Benchmarks the execution of a given function.
     * @tparam F The given function type.
     * @tparam P The lamba function parameter types.
     * @param lambda The function to be executed.
     * @param params The function's parameters.
     * @return The time spent by function's execution.
     */
    template <typename F, typename ...P>
    inline Duration<double> run(F lambda, P&&... params)
    {
        const auto start = now();
        Functor<void(P...)> {lambda}(std::forward<decltype(params)>(params)...);
        return elapsed(start);
    }
};

#endif