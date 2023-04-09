/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A base class for all generic exception types.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <exception>

#include <museqa/environment.h>

MUSEQA_BEGIN_NAMESPACE

/**
 * Represents an exception that can be thrown and propagated through the code carrying
 * a message about the error that was raised.
 * @since 1.0
 */
class exception_t : public std::exception
{
    private:
        const std::string m_msg;

    public:
        inline exception_t() = delete;
        inline exception_t(const exception_t&) = default;
        inline exception_t(exception_t&&) = default;

        /**
         * Builds a new exception instance.
         * @param msg The exception's error message.
         */
        inline explicit exception_t(const std::string& msg)
          : m_msg {msg}
        {}

        inline virtual ~exception_t() noexcept = default;

        inline exception_t& operator=(const exception_t&) = delete;
        inline exception_t& operator=(exception_t&&) = delete;

        /**
         * Returns the exception's explanatory string.
         * @return The exception message.
         */
        inline virtual const char *what() const noexcept
        {
            return m_msg.c_str();
        }
};

MUSEQA_END_NAMESPACE
