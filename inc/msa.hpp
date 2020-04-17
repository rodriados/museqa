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

namespace msa
{
    namespace watchdog
    {
        /**
         * Informs whether we're working on a report-only status.
         * @since 0.1.1
         */
        #if !__msa(runtime, cython)
            extern bool report_only;
        #else
            constexpr bool report_only = false;
        #endif

        /**
         * Notifies the watchdog about a general event.
         * @tparam T The format template arguments' types.
         * @param event The event type to send to watchdog.
         * @param fmtstr The event's format string.
         * @param args The format arguments.
         */
        template <typename ...T>
        inline void notify(const char *event, const std::string& fmtstr, T&&... args) noexcept
        {
            #if !__msa(runtime, cython)
                fmt::print("[watchdog|%s|" + fmtstr + "]\n", event, args...);
            #endif
        }

        /**
         * Prints an informative log message.
         * @tparam T The message arguments' types.
         * @param fmtstr The message format template.
         * @param args The message arguments.
         */
        template <typename ...T>
        inline void info(const std::string& fmtstr, T&&... args) noexcept
        {
            if(!report_only) notify("info", fmtstr, args...);
        }

        /**
         * Prints an error log message and halts execution.
         * @tparam T The types of message arguments.
         * @param fmtstr The message formating string.
         * @param args The message parts to be printed.
         */
        template <typename ...T>
        inline void error(const std::string& fmtstr, T&&... args) noexcept
        {
            notify("error", fmtstr, args...);
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
            notify("warning", fmtstr, args...);
        }

        /**
         * Prints a time report for given task.
         * @param taskname The name of completed task.
         * @param seconds The duration in seconds of given task.
         */
        inline void report(const char *taskname, double seconds) noexcept
        {
            onlymaster notify("report", "%s|%lf", taskname, seconds);
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
            #if !__msa(production)
                if(!report_only) notify("init", "%s|" + fmtstr, task, args...);
            #endif
        }

        /**
         * Informs the watchdog about the progress of the task being watched.
         * @param task The name of task being watched.
         * @param id The task's working node identification.
         * @param done The number of completed subtasks.
         * @param total The node's total number of subtasks.
         */
        inline void update(const char *task, size_t done, size_t total) noexcept
        {
            #if !__msa(production)
                if(!report_only) notify("update", "%s|%d|%llu|%llu", task, node::rank, done, total);
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
            #if !__msa(production)
                if(!report_only) notify("finish", "%s|" + fmtstr, task, args...);
            #endif
        }
    }
}