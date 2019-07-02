/** 
 * Multiple Sequence Alignment exception header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef EXCEPTION_HPP_INCLUDED
#define EXCEPTION_HPP_INCLUDED

#include <cstdio>
#include <string>
#include <cstring>
#include <exception>

#include "msa.hpp"
#include "utils.hpp"

#if defined(__GNUC__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wformat-security"
#endif

/**
 * Holds an error message so it can be propagated through the code.
 * @since 0.1.1
 */
class Exception : public std::exception
{
    private:
        const std::string msg;      /// The exception message.

    public:
        /**
         * Builds a new exception instance.
         * @param msg The exception message.
         */
        inline explicit Exception(const std::string& msg) noexcept
        :   msg {msg}
        {}

        /**
         * Builds a new exception instance.
         * @param fmtstr The message formating string.
         * @param args The exception message's parts.
         */
        template <typename ...T>
        inline explicit Exception(const char *fmtstr, T&&... args) noexcept
        :   Exception {fmtmsg(fmtstr, args...)}
        {}

        Exception(const Exception&) = default;
        Exception(Exception&&) = default;

        virtual ~Exception() noexcept = default;

        Exception& operator=(const Exception&) = default;
        Exception& operator=(Exception&&) = default;

        /**
         * Returns the exception's explanatory string.
         * @return The exception message.
         */
        virtual const char *what() const noexcept
        {
            return msg.c_str();
        }

    private:
        /**
         * Formats the exception string and returns it as a string.
         * @param fmtstr The exception message formatting string.
         * @param args The exception message's parts.
         * @return The formatted message.
         */
        template <typename ...T>
        inline static std::string fmtmsg(const char *fmtstr, T&&... args) noexcept
        {
            char buffer[1024 + 50 * sizeof...(T)];
            sizeof...(T)
                ? static_cast<void>(snprintf(buffer, sizeof(buffer), fmtstr, args...))
                : static_cast<void>(strcpy(buffer, fmtstr));
            return {buffer};
        }
};

/**
 * Checks whether given condition is met, and throws an exception otherwise.
 * This function acts just like an assertion, but throwing our own exception.
 * @param condition The condition that must be evaluated as true.
 * @param fmtstr The error format to be sent to an eventual thrown exception.
 * @param args The formatting arguments.
 */
template <typename ...T>
__host__ __device__ inline void enforce(bool condition, const char *fmtstr, T&&... args)
{
#ifndef msa_compile_cuda
  #ifdef msa_gcc
    if(__builtin_expect(!condition, 0))
        throw Exception(fmtstr, args...);
  #else
    if(!condition)
        throw Exception(fmtstr, args...);
  #endif
#endif
}

#if defined(__GNUC__)
  #pragma GCC diagnostic pop
#endif

#endif