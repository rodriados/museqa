/** 
 * Multiple Sequence Alignment helper functions header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef HELPER_HPP_INCLUDED
#define HELPER_HPP_INCLUDED

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

#include "colors.h"

/*
 * Defines some info macro functions. These functions should be used to
 * print out information.
 */ 
#ifndef __CUDA_ARCH__
#define info(msg, ...) printf(s_bold "[info] " s_reset msg "\n", ##__VA_ARGS__); fflush(stdout)
#else
#define info(...) // Please, do not print anything from device
#endif

/**
 * This enumeration lists all possible error types to be thrown by the application.
 * @since 0.1.alpha
 */
enum ErrorSeverity {
    ErrorSuccess = 0x00
,   ErrorWarning = 0x01
,   ErrorRuntime = 0x02
,   ErrorFatal   = 0x04
};

/**
 * This struct conveys error information so they can be easily created, returned
 * by functions and accessed anywhere.
 * @since 0.1.alpha
 */
struct Error
{
    const std::string msg;              /// The error message.
    uint8_t severity = ErrorSuccess;    /// The error severity.

    /**
     * Constructs a new error structure.
     * @param msg The error message.
     * @param severity The error severity.
     */
    Error(const std::string& msg, int severity = ErrorSuccess)
    : msg(msg), severity(severity) {}
};

/*
 * Declaring global functions.
 */
extern void errlog(Error);
extern void finalize(Error);
extern void progress(const char *, uint32_t, uint32_t, const std::string&);

#endif