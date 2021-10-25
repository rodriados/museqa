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
class exception : public std::exception
{
    private:
        const std::string m_msg;

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

MUSEQA_END_NAMESPACE
