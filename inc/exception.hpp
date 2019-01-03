/** 
 * Multiple Sequence Alignment exception header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef EXCEPTION_HPP_INCLUDED
#define EXCEPTION_HPP_INCLUDED

#include <string>
#include <exception>

/**
 * Holds an error message so it can be propagated through the code.
 * @since 0.1.1
 */
struct Exception : public std::exception
{
    std::string msg;       /// The exception message.

    /**
     * Builds a new exception instance.
     * @param msg The exception message.
     */
    explicit Exception(const std::string& msg)
    : msg(msg) {}

    /**
     * Builds a new exception instance.
     * @tparam T The format parameters' types.
     * @param fmt The message format.
     * @param args The format's parameters.
     */
    template <typename ...T>
    explicit Exception(const std::string& fmt, T... args)
    {
        msg.reserve(128);
        sprintf(msg.data(), fmt.data(), args...);
    }

    Exception(const Exception&) = default;
    Exception(Exception&&) = delete;

    Exception& operator=(const Exception&) = default;
    Exception& operator=(Exception&&) = delete;

    /**
     * Returns an explanatory string.
     * @return The explanatory string.
     */
    inline const char *what() const noexcept
    {
        return msg.data();
    }
};

#endif