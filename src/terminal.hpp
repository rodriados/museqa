/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements the software's command line input parser.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <map>
#include <string>
#include <vector>
#include <cstdint>
#include <utility>

#include "utils.hpp"
#include "functor.hpp"
#include "exception.hpp"

namespace museqa
{
    namespace terminal
    {
        /**
         * Defines a command line option, publicly available to users.
         * @since 0.1.1
         */
        struct option
        {
            std::string name;                   /// The option's name and mnemonic.
            std::vector<std::string> flags;     /// The option's matching flags.
            std::string description;            /// The option's description.
            bool is_variadic = false;           /// Does the option require arguments?
        };

        /**
         * Manages and parses parameters given by the user via command line. This
         * class shall provide an universal interface for easily retrieving parameters.
         * @since 0.1.1
         */
        class parser
        {
            protected:
                using argument_bag = std::vector<std::string>;

            protected:
                argument_bag m_positional;                      /// The list of positional arguments.
                std::map<std::string, argument_bag> m_parsed;   /// The mapping of parsed arguments.

            public:
                inline parser() = default;
                inline parser(const parser&) = default;
                inline parser(parser&&) = default;

                inline parser& operator=(const parser&) = default;
                inline parser& operator=(parser&&) = default;

                /**
                 * Checks whether an option has been parsed.
                 * @param name The name of the requested option.
                 * @return Has the option been parsed?
                 */
                inline auto has(const std::string& name) const noexcept -> bool
                {
                    const auto& it = m_parsed.find(name);
                    return it != m_parsed.end() && !it->second.empty();
                }

                /**
                 * Retrieves all arguments attached to an option.
                 * @param name The name of the requested option.
                 * @return The complete list of the option's arguments.
                 */
                inline const argument_bag all(const std::string& name) const noexcept
                {
                    return has(name) ? m_parsed.at(name) : argument_bag {};
                }

                /**
                 * Retrieves all parsed positional arguments.
                 * @return The complete list of positional arguments.
                 */
                inline const argument_bag& all() const noexcept
                {
                    return m_positional;
                }

                /**
                 * Retrieves a value associated to an option.
                 * @tparam T The type the argument must be converted to.
                 * @tparam U The default argument's type.
                 * @param name The name of option to be retrieved.
                 * @param fallback The value to be returned if none is found.
                 * @return The value of requested argument.
                 */
                template <typename T = std::string, typename U = T>
                inline auto get(const std::string& name, const U& fallback = {}) const noexcept
                -> typename std::enable_if<std::is_convertible<U, T>::value, T>::type
                try {
                    return utils::convert<T>(m_parsed.at(name).at(0));
                } catch(...) {
                    return (T) fallback;
                }

                /**
                 * Retrieves a value parsed in a positional argument.
                 * @tparam T The type the argument must be converted to.
                 * @tparam U The default argument's type.
                 * @param offset The offset of option to be retrieved.
                 * @param fallback The value to be returned if none is found.
                 * @return The value of requested positional argument.
                 */
                template <typename T = std::string, typename U = T>
                inline auto get(size_t offset, const U& fallback = {}) const noexcept
                -> typename std::enable_if<std::is_convertible<U, T>::value, T>::type
                try {
                    return utils::convert<T>(m_positional.at(offset));
                } catch(...) {
                    return (T) fallback;
                }

            friend auto parse(const std::vector<option>&, int, char **) -> parser;
        };

        /*
         * Declaration of external namespace functions.
         */
        extern auto parse(const std::vector<option>&, int, char **) -> parser;
    }
}
