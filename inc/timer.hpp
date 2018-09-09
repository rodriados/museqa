/** 
 * Multiple Sequence Alignment timer header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef TIMER_HPP_INCLUDED
#define TIMER_HPP_INCLUDED

#pragma once

#include <functional>
#include <chrono>
#include <vector>
#include <string>

/**
 * This class keeps track of time points relative to execution times of other modules.
 * @tparam R The representation type of time intervals.
 * @tparam P The period of clock ticks to use.
 * @since 0.1.alpha
 */
template <typename R = double, typename P = std::ratio<1>>
class Timer
{
    public:
        /**
         * The internal clock to be used when timing.
         * @since 0.1.alpha
         */
        using Clock = std::chrono::high_resolution_clock;

        /**
         * Stores timestamps for timing.
         * @since 0.1.alpha
         */
        using TimePoint = std::chrono::time_point<Clock>;

        /**
         * Stores a time difference.
         * @since 0.1.alpha
         */
        using TimeDiff = std::chrono::duration<R, P>;

    protected:
        TimePoint init;             /// The initial 

    public:
        /**
         * Creates a new benchmark instance.
         * @param count Projection of number of steps, so no time is lost with memory allocation.
         */
        Timer() noexcept
        :   init(now()) {}

        /**
         * Retrieves the total time elapsed since the start of exection.
         * @return The total time elapsed since step 0.
         */
        inline R elapsed() const
        {
            return TimeDiff(now() - this->init).count();
        }

        /**
         * Runs a function to be timed.
         * @param lambda The function to be executed.
         * @return The time spent by the function.
         */
        inline R run(const std::function<void()>& lambda)
        {
            TimePoint start, end;

            start = now();
            lambda();
            end = now();

            return TimeDiff(end - start).count();
        }

        /**
         * Retrieves the current time point.
         * @return The time point corresponding for now.
         */
        inline static TimePoint now()
        {
            return Clock::now();
        }
};

#endif