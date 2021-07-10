/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A base class for generic exception types and assertion implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <utility>
#include <exception>

#include <fmt/format.h>
#include <museqa/utility.hpp>
#include <museqa/environment.h>

namespace museqa
{
    /**
     * Represents an exception that can be thrown and propagated through the code
     * carrying an error message.
     * @since 1.0
     */
    class exception : public std::exception
    {
      private:
        const std::string m_msg;        /// The wrapped exception's message.

      public:
        inline exception() = delete;
        inline exception(const exception&) = default;
        inline exception(exception&&) = default;

        /**
         * Builds a new exception instance.
         * @param msg The exception's error message.
         */
        inline explicit exception(const std::string& msg)
          : m_msg {msg}
        {}

        /**
         * Builds a new exception instance.
         * @tparam T The exception's message format parameter types.
         * @param fmtstr The message formating string.
         * @param params The exception message's format parameters.
         */
        template <typename ...T>
        inline explicit exception(const std::string& fmtstr, T&&... params)
          : exception {fmt::format(fmtstr, params...)}
        {}

        inline virtual ~exception() noexcept = default;

        inline exception& operator=(const exception&) = delete;
        inline exception& operator=(exception&&) = delete;

        /**
         * Returns the exception's explanatory string.
         * @return The exception message.
         */
        inline virtual const char *what() const noexcept
        {
            return m_msg.c_str();
        }
    };

    /**
     * Checks whether given a condition is met, and throws an exception otherwise.
     * This function acts just like an assertion, but throwing our own exception.
     * @tparam E The exception type to be raised in case of error.
     * @tparam T The format string's parameter types.
     * @param condition The condition that must be evaluated as true.
     * @param fmtstr The error format to be sent to an eventual thrown exception.
     * @param params The assertion message's format parameters.
     */
    template <typename E = museqa::exception, typename ...T>
    __host__ __device__ inline void assert(bool condition, T&&... params)
    {
        static_assert(std::is_base_of<museqa::exception, E>::value, "only exception-like types are throwable");

        #if !defined(MUSEQA_UNSAFE)
            #if defined(MUSEQA_COMPILER_GNUC)
                if (__builtin_expect(!condition, 0)) {
                    throw E {std::forward<decltype(params)>(params)...};
                }
            #else
                if (!condition) {
                    throw E {std::forward<decltype(params)>(params)...};
                }
            #endif
        #endif
    }
}
