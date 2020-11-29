/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements IO operations for files and data structures.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#pragma once

#include <vector>
#include <utility>

#include "terminal.hpp"

#include "io/dumper.hpp"
#include "io/loader.hpp"

namespace museqa
{
    namespace io
    {
        /**
         * Manages and services all IO operations dependent on command line arguments,
         * flags, values and files.
         * @since 0.1.1
         */
        struct manager
        {
            terminal::parser cmd;           /// The internal command line parser instance.

            inline manager() noexcept = default;
            inline manager(const manager&) noexcept = default;
            inline manager(manager&&) noexcept = default;

            /**
             * Initializes a new IO manager with a command line parser instance.
             * @param cmd The command line parser instance to be injected.
             */
            inline explicit manager(const terminal::parser& cmd) noexcept
            :   cmd {cmd}
            {}

            /**
             * Initializes a new IO manager with a command line parser instance.
             * @param cmd The command line parser instance to be move-injected.
             */
            inline explicit manager(terminal::parser&& cmd) noexcept
            :   cmd {std::move(cmd)}
            {}

            inline manager& operator=(const manager&) noexcept = delete;
            inline manager& operator=(manager&&) noexcept = delete;

            /**
             * Loads all files compatible to one of the given type's parsers. If
             * no valid parsers are found, an empty vector is returned.
             * @tparam T The target file type to be loaded.
             * @return The list of loaded objects from files.
             */
            template <typename T>
            inline auto load() const -> std::vector<T>
            {
                static_assert(std::is_constructible<loader<T>>::value, "cannot instantiate type loader");
                static_assert(std::is_base_of<base::loader<T>, loader<T>>::value, "invalid loader type");

                auto loader = io::loader<T> {};
                auto result = std::vector<T> {};

                for(const auto& file : cmd.all())
                    if(loader.validate(file))
                        result.push_back(loader.load(file));

                return result;
            }

            /**
             * Dumps the given object into a file.
             * @tparam T The type of object to be dumped.
             * @param obj The object to be dumped into a file.
             * @param filename The name of the file to dumpt object into.
             * @return Has the object been successfully dumped?
             */
            template <typename T>
            inline auto dump(const T& obj, const std::string& filename) const -> bool
            {
                static_assert(std::is_constructible<dumper<T>>::value, "cannot instantiate type dumper");
                static_assert(std::is_base_of<base::dumper<T>, dumper<T>>::value, "invalid dumper type");

                auto dumper = io::dumper<T> {};
                return dumper.dump(obj, filename);
            }

            /**
             * Initializes a new IO manager directly from command line arguments.
             * @param options The list of available command line options.
             * @param argc The number of command line arguments.
             * @param argv The command line arguments.
             * @return The newly created IO manager instance.
             */
            inline static auto make(const std::vector<terminal::option>& options, int argc, char **argv) -> manager
            {
                auto cmd = terminal::parse(options, argc, argv);
                return manager {std::move(cmd)};
            }
        };
    }
}
