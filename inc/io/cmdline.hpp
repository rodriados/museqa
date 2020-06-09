/** 
 * Multiple Sequence Alignment command line header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2020 Rodrigo Siqueira
 */
#pragma once

#include <map>
#include <string>
#include <vector>
#include <utility>

#include <exception.hpp>

namespace msa
{
    namespace io
    {
        /**
         * Manages and parses parameters given by the user via command line. This class
         * shall provide an universal interface for easily retrieving such parameters.
         * @since 0.1.1
         */
        class cmdline
        {
            public:
                /**
                 * The type to which options must be identified by.
                 * @since 0.1.1
                 */
                using code = uint32_t;

                /**
                 * Defines a command line option, publicly available to users.
                 * @since 0.1.1
                 */
                struct option
                {
                    code opcode = 0;                        /// The option's code mnemonic.
                    std::vector<std::string> flags;         /// The option's matching flags.
                    std::string description;                /// The option's brief description.
                    bool variadic = false;                  /// Does the option require an argument?
                };

            protected:
                std::map<code, std::string> m_parsed;       /// The list of parsed option results.
                std::vector<std::string> m_posargs;         /// The list of positional arguments.

            public:
                /**
                 * Initializes a new command line service instance. As a command line
                 * parser and manager, we shall parse at startup
                 * @param options The list of available options to be parsed.
                 * @param argc The number of command line arguments.
                 * @param argv The command line arguments.
                 */
                inline explicit cmdline(const std::vector<option>& options, int argc, char **argv)
                {
                    parse(options, argc, argv);
                }

                /**
                 * Checks whether an option has been parsed.
                 * @param code The code of the requested option.
                 * @return Has the option been parsed?
                 */
                inline auto has(const code& opcode) const noexcept -> bool
                {
                    const auto& it = m_parsed.find(opcode);
                    return it != m_parsed.end() && !it->second.empty();
                }

                /**
                 * Retrieves the value associated to an option.
                 * @tparam T The type the argument must be converted to.
                 * @param opcode The code of option to be retrieved.
                 * @param fallback The value to be returned if none is found.
                 * @return The value of requested argument.
                 */
                template <typename T = std::string>
                inline auto get(const code& opcode, const T& fallback = {}) const noexcept -> T
                {
                    return has(opcode) ? convert<T>(m_parsed.at(opcode)) : fallback;
                }

                /**
                 * Returns the complete list of positional arguments.
                 * @return The list of positional arguments.
                 */
                inline auto positional() const noexcept -> const std::vector<std::string>&
                {
                    return m_posargs;
                }

                /**
                 * Retrieves the value received by a positional argument.
                 * @tparam T The type the argument must be converted to.
                 * @param offset The offset of requested argument.
                 * @param fallback The value to be returned if none is found.
                 * @return The value of requested argument.
                 */
                template <typename T = std::string>
                inline auto positional(ptrdiff_t offset, const T& fallback = {}) const noexcept -> T
                {
                    return m_posargs.size() > size_t(offset) ? convert<T>(m_posargs[offset]) : fallback;
                }

            protected:
                auto parse(const std::vector<option>&, int, char **) -> void;

            private:
                /**
                 * Converts the given argument to a general type.
                 * @tparam T The target type to convert to.
                 * @param arg The argument value to be converted.
                 * @return The converted value from given argument.
                 */
                template <typename T>
                inline static auto convert(const std::string& arg)
                -> typename std::enable_if<std::is_convertible<std::string, T>::value, T>::type
                {
                    return T (arg);
                }

                /**
                 * Converts the given argument to an integral type.
                 * @tparam T The target type to convert to.
                 * @param arg The argument value to be converted.
                 * @return The converted value from given argument.
                 * @throw exception Error detected during operation.
                 */
                template <typename T>
                inline static auto convert(const std::string& arg)
                -> typename std::enable_if<std::is_integral<T>::value, T>::type
                try {
                    return static_cast<T>(std::stoull(arg));
                } catch(const std::invalid_argument&) {
                    throw exception {"unable to convert argument to integer '%s'", arg};
                } catch(const std::out_of_range&) {
                    throw exception {"argument numeric value out of range '%s'", arg};
                }

                /**
                 * Converts the given argument to a floating point type.
                 * @tparam T The target type to convert to.
                 * @param arg The argument value to be converted.
                 * @return The converted value from given argument.
                 * @throw exception Error detected during operation.
                 */
                template <typename T>
                inline static auto convert(const std::string& arg)
                -> typename std::enable_if<std::is_floating_point<T>::value, T>::type
                try {
                    return static_cast<T>(std::stold(arg));
                } catch(const std::invalid_argument&) {
                    throw exception {"unable to convert argument to floating point '%s'", arg};
                } catch(const std::out_of_range&) {
                    throw exception {"argument numeric value out of range '%s'", arg};
                }
        };
    }
}
