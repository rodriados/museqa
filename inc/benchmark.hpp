/** 
 * Multiple Sequence Alignment benchmark header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef BENCHMARK_HPP_INCLUDED
#define BENCHMARK_HPP_INCLUDED

#pragma once

#include <chrono>
#include <vector>
#include <string>

/**
 * This class keeps track of time points relative to execution times of other modules.
 * @since 0.1.alpha
 */
class Benchmark
{
    private:
        typedef std::chrono::high_resolution_clock Clock;
        typedef std::chrono::time_point<Clock> TimePoint;
        typedef std::chrono::duration<double> Duration;

    protected:
        std::vector<TimePoint> run;                 /// The time point runs.

    public:
        /**
         * Creates a new benchmark instance.
         * @param count Projection of number of steps, so no time is lost with memory allocation.
         */
        Benchmark(int count = 5)
        {
            this->run.reserve(count + 1);
            this->step();
        }

        /**
         * Retrieves the number of steps registered in the container.
         * @return The number of time point runs registered.
         */
        inline int getCount() const
        {
            return this->run.size() - 1;
        }

        /**
         * Retrieves the time elapsed by a given step.
         * @param id The id of requested step.
         * @param sinceStart Should the time be counted since start?
         * @return The time elapsed by step.
         */
        inline auto getStep(int id, bool sinceStart = false) const
        -> decltype(std::declval<Duration>().count())
        {
            return sinceStart
                ? Duration(this->run[id + 1] - this->run[0]).count()
                : Duration(this->run[id + 1] - this->run[id]).count();
        }

        /**
         * Retrieves the total time elapsed since the start of exection.
         * @return The total time elapsed since step 0.
         */
        inline auto elapsed() const
        -> decltype(std::declval<Duration>().count())
        {
            return Duration(now() - this->run[0]).count();
        }

        /**
         * Registers a new time point step.
         * @return The time point set to the step.
         */
        inline TimePoint step()
        {
            TimePoint point = now();

            this->run.push_back(point);
            return point;
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