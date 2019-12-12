/**
 * Multiple Sequence Alignment using hybrid parallel computing
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <cstddef>
#include <cstdint>

#include <environment.h>

#include <node.hpp>
#include <utils.hpp>
#include <format.hpp>
#include <exception.hpp>

namespace msa
{
    #if !__msa(runtime, cython)
        /**
         * Halts the whole software's execution and exits with given code.
         * @param code The exit code.
         */
        [[noreturn]] extern void halt(uint8_t = 0) noexcept;
    #endif

    namespace watchdog
    {
        /**
         * Prints an informative log message.
         * @tparam T The message arguments' types.
         * @param fmtstr The message format template.
         * @param args The message arguments.
         */
        template <typename ...T>
        inline void info(const std::string& fmtstr, T&&... args) noexcept
        {
            #if !__msa(runtime, cython)
                fmt::print("[_WTCHDG_|info|" + fmtstr + "]\n", args...);
            #endif
        }

        /**
         * Prints an error log message and halts execution.
         * @tparam T The types of message arguments.
         * @param fmtstr The message formating string.
         * @param args The message parts to be printed.
         */
        template <typename ...T>
        inline void error(const std::string& fmtstr, T&&... args)
        {
            #if !__msa(runtime, cython)
                fmt::print("[_WTCHDG_|error|" + fmtstr + "]\n", args...);
                msa::halt(1);
            #else
                throw exception {fmtstr, args...};
            #endif
        }

        /**
         * Prints a warning log message.
         * @tparam T The types of message arguments.
         * @param fmtstr The message formating string.
         * @param args The message parts to be printed.
         */
        template <typename ...T>
        inline void warning(const std::string& fmtstr, T&&... args) noexcept
        {
            #if !__msa(runtime, cython)
                fmt::print("[_WTCHDG_|warning|" + fmtstr + "]\n", args...);
            #endif
        }

        /**
         * Informs the watchdog about a new task to be watched.
         * @param task The name of task to be watched.
         * @param fmtstr The message formatting string.
         * @param args The message parts to be printed.
         */
        template <typename ...T>
        inline void init(const char *task, const std::string& fmtstr, T&&... args) noexcept
        {
            #if !__msa(runtime, cython)
                fmt::print("[_WTCHDG_|init|%s|" + fmtstr + "]\n", task, args...);
            #endif
        }

        /**
         * Informs the watchdog about the progress of the task being watched.
         * @param task The name of task being watched.
         * @param id The task's working node identification.
         * @param done The number of completed subtasks.
         * @param total The node's total number of subtasks.
         */
        inline void update(const char *task, int id, size_t done, size_t total)
        {
            #if !__msa(runtime, cython)
                fmt::print("[_WTCHDG_|update|%s|%d|%llu|%llu]\n", task, id, done, total);
            #endif
        }

        /**
         * Informs the watchdog about the completion of the watched task.
         * @param task The name of task to be watched.
         * @param fmtstr The message formatting string.
         * @param args The message parts to be printed.
         */
        template <typename ...T>
        inline void finish(const char *task, const std::string& fmtstr, T&&... args) noexcept
        {
            #if !__msa(runtime, cython)
                fmt::print("[_WTCHDG_|finish|%s|" + fmtstr + "]\n", task, args...);
            #endif
        }

        /**
         * Prints a time report for given task.
         * @param taskname The name of completed task.
         * @param seconds The duration in seconds of given task.
         */
        inline void report(const char *taskname, double seconds) noexcept
        {
            #if !__msa(runtime, cython)
                onlymaster watchdog::info("<bold green>%s</> done in <bold>%lf</> seconds", taskname, seconds);
            #endif
        }
    }
}