/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Timer middleware for benchmarking pipeline modules execution.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <vector>

#include <museqa/environment.h>
#include <museqa/benchmark.cuh>
#include <museqa/pipeline.hpp>

#include <museqa/utility/tuple.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace heuristic::middleware
{
    /**
     * The pipeline middleware for measuring the time elapsed during the execution
     * of the wrapped modules. The measured duration is stored within the module.
     * If executed more than once, the duration is stored in different runs.
     * @tparam M The list of modules to measure execution time.
     * @since 1.0
     */
    template <typename ...M>
    class timer_t : public pipeline::middleware_t<M...>
    {
        private:
            typedef pipeline::middleware_t<M...> underlying_t;

        protected:
            mutable std::vector<double> m_duration;

        public:
            using underlying_t::middleware_t;

            /**
             * Executes the core logic of all modules referenced by the middleware,
             * while the elapsed execution time is measured.
             * @param pipe The pipeline's transitive state instance.
             */
            inline void run(pipeline::pipe_t& pipe) const override
            {
                const auto lambda = [&]{ underlying_t::run(pipe); };
                const auto duration = utility::last(benchmark::run(lambda));
                m_duration.push_back(duration);
            }

            /**
             * Retrieves the time measurement for the referenced modules' execution.
             * @param run The index of the requested execution run.
             * @return The total amount of time elapsed during the modules' execution.
             */
            inline auto duration(size_t run = 0) const -> double
            {
                return run < m_duration.size()
                    ? m_duration[run]
                    : 0.f;
            }
    };

    /*
     * Deduction guides for the pipeline's timer middleware.
     * @since 1.0
     */
    template <typename ...M> timer_t(M&&...) -> timer_t<M...>;
}

MUSEQA_END_NAMESPACE
