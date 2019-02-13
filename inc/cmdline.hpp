/** 
 * Multiple Sequence Alignment command line header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef CMDLINE_HPP_INCLUDED
#define CMDLINE_HPP_INCLUDED

#include <utility>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>

namespace cmdline
{
    /**
     * Stores all information about a given option available from the command
     * line. There should be an instance for each option available.
     * @since 0.1.1
     */
    class Option
    {
        protected:
            std::string sname;              /// The option's short name.
            std::string lname;              /// The option's long name.
            bool variadic;                  /// Is the option variadic?
            bool required;                  /// Is the option required?

        public:
            Option() noexcept = default;
            Option(const Option&) = default;
            Option(Option&&) = default;

            /**
             * Builds an option from its names and description.
             * @param sname The option's short name.
             * @param lname The option's long name.
             * @param (ignored) The option's description.
             * @param flag The option's flags.
             */
            inline Option
                (   const std::string& sname
                ,   const std::string& lname
                ,   const std::string& //description
                ,   const bool variadic = false
                ,   const bool required = false         )
                noexcept
            :   sname(sname)
            ,   lname(lname)
            ,   variadic(variadic)
            ,   required(required)
            {}

            Option& operator=(const Option&) = default;
            Option& operator=(Option&&) = default;

            /**
             * Gets the option's short name.
             * @return The retrieved option short name.
             */
            inline const std::string& getSname() const
            {
                return sname;
            }

            /**
             * Gets the option's long name.
             * @return The retrieved option long name.
             */
            inline const std::string& getLname() const
            {
                return lname;
            }

            /**
             * Checks whether the given string correspond to this option.
             * @param given The option name to check.
             * @return Is this the requested option?
             */
            inline bool is(const std::string& given) const
            {
                return given == sname
                    || given == lname;
            }

            /**
             * Checks whether the option is required.
             * @return Is the option required?
             */
            inline bool isRequired() const
            {
                return required;
            }

            /**
             * Checks whether the option is variadic.
             * @return Is the option variadic.
             */
            inline bool isVariadic() const
            {
                return variadic;
            }

            /**
             * Checks whether the option is empty or unknown.
             * @return Is the option unknown?
             */
            inline bool isUnknown() const
            {
                return sname.empty()
                    && lname.empty();
            }
    };

    /**
     * Parses the parameters given from the command line, and stores them so it can
     * be easily retrieved when needed.
     * @since 0.1.1
     */
    class Parser
    {
        private:
            std::vector<std::string> required;          /// The list of required options.
            std::map<std::string, Option> options;      /// The map of available command line options.

        protected:
            std::string appname;                        /// The name used by the application.
            std::vector<std::string> positional;        /// The list of positional arguments.
            std::map<std::string, std::string> values;  /// The map of parsed option arguments values.

        public:
            Parser() noexcept = default;
            Parser(const Parser&) = default;
            Parser(Parser&&) = default;

            /**
             * Creates a new parser instance and sets the available options it may parse.
             * @param options The list of available options.
             */
            inline Parser(const std::vector<Option>& options) noexcept
            {
                init(options);
            }

            Parser& operator=(const Parser&) = default;
            Parser& operator=(Parser&&) = default;

            /**#@+
             * Retrieves the value received by a named argument.
             * @tparam T The type the argument must be converted to.
             * @param name The name of the requested argument.
             * @param fallback The value to be returned if none is found.
             * @return The value of requested argument.
             */
            template <typename T = std::string>
            inline auto get(const std::string& name, const T& fallback = {}) const
            -> typename std::enable_if<!std::is_arithmetic<T>::value, T>::type
            {                
                static_assert(std::is_convertible<std::string, T>::value, "Cannot convert to requested type");
                const auto& pair = values.find(name);

                return pair != values.end()
                    ? static_cast<T>(pair->second)
                    : fallback;
            }

            template <typename T>
            inline auto get(const std::string& name, const T& fallback = {}) const
            -> typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, T>::type
            {
                return has(name)
                    ? static_cast<T>(strtoll(get(name).c_str(), nullptr, 0))
                    : fallback;
            }

            template <typename T>
            inline auto get(const std::string& name, const T& fallback = {}) const
            -> typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value, T>::type
            {
                return has(name)
                    ? static_cast<T>(strtoull(get(name).c_str(), nullptr, 0))
                    : fallback;
            }

            template <typename T>
            inline auto get(const std::string& name, const T& fallback = {}) const
            -> typename std::enable_if<std::is_floating_point<T>::value, T>::type
            {
                return has(name)
                    ? static_cast<T>(strtold(get(name).c_str(), nullptr))
                    : fallback;
            }
            /**#@-*/

            /**
             * Informs the name used to start the application.
             * @return The application name used.
             */
            inline const std::string& getAppname() const
            {
                return appname;
            }

            /**
             * Informs the number of parsed arguments.
             * @return The number of parsed arguments.
             */
            inline size_t getCount() const
            {
                return values.size();
            }

            /**
             * Returns the whole list of positional arguments.
             * @return The list of parsed positional arguments.
             */
            inline const std::vector<std::string>& getPositional() const
            {
                return positional;
            }

            /**
             * Checks whether an argument exists.
             * @param argname The name of the requested argument.
             * @return Does the argument exist?
             */
            inline bool has(const std::string& argname) const
            {
                return values.find(argname) != values.end();
            }

            void init(const std::vector<Option>&);
            void parse(int, char **);

        protected:
            const Option& find(const std::string&) const;
    };

    extern Parser parser;

    /**
     * Retrieves the value received by a named argument.
     * @tparam T The type the argument must be converted to.
     * @param name The name of the requested argument.
     * @param fallback The value to be returned if none is found.
     * @return The value of requested argument.
     */
    template <typename T>
    inline T get(const std::string& name, const T& fallback = {})
    {
        return parser.get<T>(name, fallback);
    }

    /**
     * Informs the name used to start the application.
     * @return The application name used.
     */
    inline const std::string& getAppname()
    {
        return parser.getAppname();
    }

    /**
     * Informs the number of parsed arguments.
     * @return The number of parsed arguments.
     */
    inline size_t getCount()
    {
        return parser.getCount();
    }

    /**
     * Returns the whole list of positional arguments.
     * @return The list of parsed positional arguments.
     */
    inline const std::vector<std::string>& getPositional()
    {
        return parser.getPositional();
    }

    /**
     * Checks whether an argument exists.
     * @param argname The name of the requested argument.
     * @return Does the argument exist?
     */
    inline bool has(const std::string& argname)
    {
        return parser.has(argname);
    }

    /**
     * Initializes the command line arguments.
     * @param config The options available.
     */
    inline void init(const std::vector<Option>& options)
    {
        parser.init(options);
    }

    /**
     * Parses the command line arguments.
     * @param argc Number of arguments sent by command line.
     * @param argv The arguments sent by command line.
     */
    inline void parse(int argc, char **argv)
    {
        parser.parse(argc, argv);
    }
};

#endif