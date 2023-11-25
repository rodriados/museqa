/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Timer middleware for benchmarking pipeline modules execution.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

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
     * @tparam M The list of modules to measure execution time.
     * @since 1.0
     */
    template <typename ...M>
    class timer_t : public pipeline::middleware_t<M...>
    {
        private:
            typedef pipeline::middleware_t<M...> underlying_t;

        protected:
            mutable double m_duration = 0.f;

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
                m_duration = utility::last(benchmark::run(lambda));
            }

            /**
             * Retrieves the time measurement for the referenced modules' execution.
             * @return The total amount of time elapsed during the modules' execution.
             */
            inline auto duration() const -> double
            {
                return m_duration;
            }
    };

    /*
     * Deduction guides for the pipeline's timer middleware.
     * @since 1.0
     */
    template <typename ...M> timer_t(M&&...) -> timer_t<M...>;
}

MUSEQA_END_NAMESPACE
