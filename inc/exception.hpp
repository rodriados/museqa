/** 
 * Multiple Sequence Alignment exception header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <utility>
#include <exception>

#include <utils.hpp>
#include <format.hpp>

namespace msa
{
    /**
     * Holds an error message so it can be propagated through the code.
     * @since 0.1.1
     */
    class exception : public std::exception
    {
        private:
            const std::string m_msg;    /// The exception's informative message.

        public:
            inline exception() = delete;
            inline exception(const exception&) = default;
            inline exception(exception&&) = default;

            /**
             * Builds a new exception instance.
             * @param msg The exception's informative message.
             */
            inline explicit exception(const std::string& msg) noexcept
            :   m_msg {msg}
            {}

            /**
             * Builds a new exception instance.
             * @tparam T The message argument types.
             * @param fmtstr The message formating string.
             * @param args The exception message's format arguments.
             */
            template <typename ...T>
            inline explicit exception(const std::string& fmtstr, T&&... args) noexcept
            :   exception {fmt::format(fmtstr, args...)}
            {}

            virtual ~exception() noexcept = default;

            inline exception& operator=(const exception&) = delete;
            inline exception& operator=(exception&&) = delete;

            /**
             * Returns the exception's explanatory string.
             * @return The exception message.
             */
            virtual const char *what() const noexcept
            {
                return m_msg.c_str();
            }
    };

    /**
     * Checks whether given condition is met, and throws an exception otherwise.
     * This function acts just like an assertion, but throwing our own exception.
     * @tparam E The exception type to be raised in case of error.
     * @tparam T The format string's parameter types.
     * @param condition The condition that must be evaluated as true.
     * @param fmtstr The error format to be sent to an eventual thrown exception.
     * @param args The formatting arguments.
     */
    template <typename E = exception, typename ...T>
    __host__ __device__ inline void enforce(bool condition, T&&... args)
    {
        static_assert(std::is_base_of<exception, E>::value, "only exception types are throwable");

        #if !defined(__msa_runtime_device) && !defined(__msa_production)
            #if defined(__msa_compiler_gnuc)
                if(__builtin_expect(!condition, 0)) {
                    throw E {std::forward<decltype(args)>(args)...};
                }
            #else
                if(!condition) {
                    throw E {std::forward<decltype(args)>(args)...};
                }
            #endif
        #endif
    }
}
