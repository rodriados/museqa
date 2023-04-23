/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Helper functions for lambda calls.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2023-present Rodrigo Siqueira
 */
#pragma once

#include <utility>

#include <museqa/environment.h>

MUSEQA_BEGIN_NAMESPACE

namespace utility::detail
{
    /**
     * Seamlessly calls a callable depending on whether its a free function, lambda
     * or a bound member function pointer.
     * @tparam F The callable type.
     * @tparam M The call's first parameter and possible member instance.
     * @tparam A The remaining call's arguments types.
     * @param lambda The callable instance to be invoked.
     * @param member The possible member instance to bind the callable to.
     * @param args The remaining call's arguments.
     * @return The call's result.
     */
    template <typename F, typename M, typename ...A>
    __host__ __device__ inline constexpr decltype(auto) polymorphic_call(
        const F& lambda
      , M&& member
      , A&&... args
    ) {
        if constexpr (std::is_member_function_pointer<F>::value) {
            return (member.*lambda)(std::forward<decltype(args)>(args)...);
        } else {
            return lambda(
                std::forward<decltype(member)>(member)
              , std::forward<decltype(args)>(args)...
            );
        }
    }
}

MUSEQA_END_NAMESPACE
