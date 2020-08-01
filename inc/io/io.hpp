/**
 * Multiple Sequence Alignment input and output header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020 Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <vector>
#include <utility>

#include <io/dumper.hpp>
#include <io/loader.hpp>
#include <io/cmdline.hpp>

namespace msa
{
    namespace io
    {
        /**
         * Defines a command line option, publicly available to users.
         * @since 0.1.1
         */
        using option = cmdline::option;

        /**
         * Manages and services all IO operations dependent on command line arguments,
         * flags, values and files.
         * @since 0.1.1
         */
        class service
        {
            protected:
                cmdline m_cmd;              /// The internal command line manager.

            public:
                inline service() noexcept = delete;
                inline service(const service&) = default;
                inline service(service&&) = default;

                /**
                 * Initializes a new IO service with a command line manager instance.
                 * @param cmd The command line manager instance to be injected.
                 */
                inline explicit service(const cmdline& cmd) noexcept
                :   m_cmd {cmd}
                {}

                /**
                 * Initializes a new IO service with a command line manager instance.
                 * @param cmd The command line manager instance to be moved.
                 */
                inline explicit service(cmdline&& cmd) noexcept
                :   m_cmd {std::move(cmd)}
                {}

                inline service& operator=(const service&) = delete;
                inline service& operator=(service&&) = delete;

                /**
                 * Checks whether a command line option has been parsed.
                 * @param code The code of the requested option.
                 * @return Has the option been parsed?
                 */
                inline auto has(const cmdline::code& opcode) const noexcept -> bool
                {
                    return m_cmd.has(opcode);
                }

                /**
                 * Retrieves the value associated to a command line option.
                 * @tparam T The type the argument must be converted to.
                 * @param opcode The code of option to be retrieved.
                 * @param fallback The value to be returned if none is found.
                 * @return The value of requested argument.
                 */
                template <typename T = std::string>
                inline auto get(const cmdline::code& opcode, const T& fallback = {}) const noexcept -> T
                {
                    return m_cmd.template get<T>(opcode, fallback);
                }

                /**
                 * Informs the total number of files passed as command line arguments.
                 * @return The number of files available for parsing.
                 */
                inline auto filecount() const noexcept -> size_t
                {
                    return m_cmd.positional().size();
                }

                /**
                 * Loads all files compatible to one of the given type's parsers.
                 * If no valid parsers are found, an empty vector is returned.
                 * @tparam T The target file type to be loaded.
                 * @return The list of loaded objects from files.
                 */
                template <typename T>
                inline auto load() const -> std::vector<T>
                {
                    static_assert(std::is_constructible<loader<T>>::value, "cannot instantiate type loader");
                    static_assert(std::is_base_of<base_loader<T>, loader<T>>::value, "invalid loader type");

                    auto loader = io::loader<T> {};
                    auto result = std::vector<T> {};

                    for(const auto& file : m_cmd.positional())
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
                    static_assert(std::is_base_of<base_dumper<T>, dumper<T>>::value, "invalid dumper type");

                    auto dumper = io::dumper<T> {};
                    return dumper.dump(obj, filename);
                }

                /**
                 * Creates a new IO service directly from command line arguments.
                 * @param options The list of available command line options.
                 * @param argc The number of command line arguments.
                 * @param argv The command line arguments.
                 * @return The newly created IO service instance.
                 */
                static auto make(const std::vector<option>& options, int argc, char **argv) -> service
                {
                    auto cmd = cmdline {options, argc, argv};
                    return service {std::move(cmd)};
                }
        };
    }
}
