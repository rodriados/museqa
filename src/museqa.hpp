/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The software's environment configuration and helper functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <cstddef>
#include <cstdint>

#include "environment.h"

#include "node.hpp"
#include "utils.hpp"
#include "format.hpp"

namespace museqa
{
    /**
     * Lists all different environments in which the code may be executed on. Different
     * code behaviours may arise depending on the environment in which compiled.
     * @since 0.1.1
     */
    enum class env
    {
        debug = 1,
        testing = 2,
        production = 3,
        dev = 4,
    };

    /**
     * Stores globally relevant state. These values may be gathered during runtime
     * and may affect the software's execution behaviour.
     * @since 0.1.1
     */
    struct state
    {
        bool report_only = false;       /// Should only time report messages be printed?
        bool mpi_running = false;       /// Is the MPI runtime running?
        bool use_multigpu = false;      /// Should MPI nodes use more than one GPU?
        bool use_devices = false;       /// Indicates whether devices should be used by default.
        int local_devices = 0;          /// The number of GPU devices available on node.

        const env environment;          /// The execution runtime's environment.

        /**
         * Initializes the global state with the runtime environment configuration.
         * @param environment The environment to which the code is compiled to.
         */
        constexpr inline state(env environment) noexcept
        :   environment {environment}
        {}
    };

    /**
     * The global state instance.
     * @since 0.1.1
     */
    #if defined(__museqa_runtime_cython)
        constexpr state global_state = {env::testing};
    #else
        extern state global_state;
    #endif

    namespace watchdog
    {
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
            #if !defined(__museqa_runtime_cython)
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
            if(!global_state.report_only)
                notify("info", fmtstr, args...);
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
            #if !defined(__museqa_production)
                if(!global_state.report_only)
                    notify("init", "%s|" + fmtstr, task, args...);
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
            #if !defined(__museqa_production)
                if(!global_state.report_only)
                    notify("finish", "%s|" + fmtstr, task, args...);
            #endif
        }
    }
}
