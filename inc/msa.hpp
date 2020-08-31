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
    /**
     * Lists all different environments in which the code may find itself. Different
     * code behaviours may arise depending on the environment in which compiled.
     * @since 0.1.1
     */
    enum class environment
    {
        debug = 1,
        testing = 2,
        production = 3,
        dev = 4,
    };

    /**
     * Stores globally relevant state. These values may be gathered during runtime
     * and may affect the software's execution.
     * @since 0.1.1
     */
    struct state
    {
        const environment env;
        bool report_only = false;
        bool mpi_running = false;
        bool use_multigpu = false;
        int devices_available = 0;

        /**
         * Initializes the global state with the environment configuration.
         * @param env The environment in which the code is compiled.
         */
        constexpr inline state(environment env) noexcept
        :   env {env}
        {}
    };

    namespace cli
    {
        /**
         * Lists every command line flags available to users. These command line
         * options do not accept arguments.
         * @since 0.1.1
         */
        enum flag : uint32_t
        {
            multigpu    = 101
        ,   report      = 102
        };

        /**
         * Lists all command line options available to users. These enumerations
         * will always be used when a command line argument is required.
         * @since 0.1.1
         */
        enum argument : uint32_t
        {
            scoring     = 201
        ,   pairwise    = 202
        ,   phylogeny   = 203
        ,   gpuid       = 204
        };
    }

    /**
     * The global state instance.
     * @since 0.1.1
     */
    #if defined(__msa_runtime_cython)
        constexpr state global_state = {environment::testing};
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
            #if !defined(__msa_runtime_cython)
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
            #if !defined(__msa_production)
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
            #if !defined(__msa_production)
                if(!global_state.report_only)
                    notify("finish", "%s|" + fmtstr, task, args...);
            #endif
        }
    }
}
