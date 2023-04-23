/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A functor extension to perform timing benchmark analysis.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <chrono>
#include <utility>

#include <museqa/environment.h>

#if MUSEQA_INCLUDE_DEVICE_CODE
  #include <cuda/std/chrono>
#endif

#include <museqa/utility.hpp>
#include <museqa/utility/tuple.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace benchmark
{
    /**
     * Wraps the given functor-like object into a timing scope, runs it with the
     * given parameters and returns, alongside the functor's resulting value, the
     * total time taken by the functor to execute and return.
     * @tparam D The period to which the benchmark must measure time relative to.
     * @tparam C The clock type to be used for measuring the time taken by the functor.
     * @tparam F The type of the functor to be benchmarked.
     * @tparam P The types of parameters to execute the functor with.
     * @param functor The functor to be wrapped and have its execution time measured.
     * @param params The parameters to execute the wrapped functor with.
     * @return The functor's return value and its execution time.
     */
    template <
      #if !MUSEQA_RUNTIME_DEVICE
          typename D = std::chrono::seconds
        , typename C = typename std::conditional<
              std::chrono::high_resolution_clock::is_steady
            , std::chrono::high_resolution_clock
            , std::chrono::steady_clock
          >::type
      #else
          typename D = ::cuda::std::chrono::seconds
        , typename C = ::cuda::std::chrono::high_resolution_clock
      #endif
      , typename F
      , typename ...P
    >
    __host__ __device__ inline auto run(F&& functor, P&&... params) -> decltype(auto)
    {
        const auto start = C::now();

        // First and foremost, let us discover what is the type returned by the
        // functor when given the parameters we have received. This is essential
        // to figure out what type we must also return.
        using result_t = decltype(
            utility::invoke(functor, std::forward<decltype(params)>(params)...)
        );

        // Now that the resulting type of the functor is well known, we can allocate
        // a variable to the its return value. If the functor happens to not return
        // anything, then we must return nothing.
        typename std::conditional<
                std::is_void<result_t>::value
              , museqa::nothing_t
              , result_t
            >::type result;

        // Run the functor with the given parameters and store its result if it
        // produces any. Otherwise, ignore the result and simply execute it. The
        // functor invokation and its result's copying operation must be the only
        // computation within the benchmark scope.
        if constexpr (!std::is_void<result_t>::value) {
            result = utility::invoke(functor, std::forward<decltype(params)>(params)...);
        } else {
            utility::invoke(functor, std::forward<decltype(params)>(params)...);
        }

        // Now that the functor has already been executed, we must build a duration
        // type known at the current runtime and with the requested time period.
      #if !MUSEQA_RUNTIME_DEVICE
        using duration_t = std::chrono::duration<double, typename D::period>;
      #else
        using duration_t = ::cuda::std::chrono::duration<double, typename D::period>;
      #endif

        // Finally, we can calculate the total time spent by the functor to perform
        // its computation. At this point, we are ready to return the results.
        const duration_t duration = C::now() - start;

        return utility::tuple_t(result, duration.count());
    }
}

MUSEQA_END_NAMESPACE
