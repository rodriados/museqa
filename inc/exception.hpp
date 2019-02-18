/** 
 * Multiple Sequence Alignment exception header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef EXCEPTION_HPP_INCLUDED
#define EXCEPTION_HPP_INCLUDED

#include <string>
#include <sstream>
#include <exception>

#include "msa.hpp"

/**
 * Holds an error message so it can be propagated through the code.
 * @since 0.1.1
 */
struct Exception : public std::exception
{
    std::string msg;        /// The exception message.

    /**
     * Builds a new exception instance.
     * @param msg The exception message.
     */
    explicit Exception(const std::string& msg)
    :   msg {msg}
    {}

    /**
     * Builds a new exception instance.
     * @param args The exception message's parts.
     */
    template <typename ...T>
    explicit Exception(const T&... args)
    {
        std::stringstream ss;
        msa::log(ss, args...);
        msg = ss.str();
    }

    Exception(const Exception&) = default;
    Exception(Exception&&) = default;

    virtual ~Exception() noexcept = default;

    Exception& operator=(const Exception&) = default;
    Exception& operator=(Exception&&) = default;

    /**
     * Returns an explanatory string.
     * @return The explanatory string.
     */
    inline const char *what() const noexcept { return msg.data(); }
};

#endif