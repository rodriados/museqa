/** 
 * Multiple Sequence Alignment command line header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef CMDLINE_HPP_INCLUDED
#define CMDLINE_HPP_INCLUDED

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <map>

namespace cmdline
{
    /**
     * Option flags enumeration.
     * @since 0.1.1
     */
    enum Flag
    {
        required = 0x01
    ,   variadic = 0x02
    };

    /**
     * Stores all information about a given option available from the command
     * line. There should be an instance for each option available.
     * @since 0.1.1
     */
    class Option
    {
        protected:
            std::string sname;            /// The option's short name.
            std::string lname;            /// The option's long name.
            uint8_t flag = 0;                   /// Is the option required?

        public:
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
                ,   const uint8_t flag = 0             ) noexcept
            :   sname(sname)
            ,   lname(lname)
            ,   flag(flag) {}

            Option() noexcept = default;
            Option(const Option&) = default;
            Option(Option&&) = default;

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
                return flag & required;
            }

            /**
             * Checks whether the option is variadic.
             * @return Is the option variadic.
             */
            inline bool isVariadic() const
            {
                return flag & variadic;
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
            std::map<std::string, std::string> args;    /// The map of parsed option arguments.

        public:
            /**
             * Creates a new parser instance and sets the available options it may parse.
             * @param options The list of available options.
             */
            inline Parser(const std::vector<Option>& options) noexcept
            {
                init(options);
            }

            Parser() noexcept = default;
            Parser(const Parser&) = default;
            Parser(Parser&&) = default;

            Parser& operator=(const Parser&) = default;
            Parser& operator=(Parser&&) = default;

            /**
             * Retrieves the value received by a named argument.
             * @param argname The name of the requested argument.
             * @param fallback The value to be returned if none is found.
             * @return The value of requested argument.
             */
            inline const std::string& get(const std::string& name, const std::string& fallback = "") const
            {
                const auto& value = args.find(name);

                return value != args.end()
                    ? value->second
                    : fallback;
            }

            /**
             * Retrieves the value of a positional argument.
             * @param index The requested argument index.
             * @param fallback The value to be returned if none is found.
             * @return The value of requested argument.
             */
            inline const std::string& get(uint16_t index, const std::string& fallback = "") const
            {
                return positional.size() > index
                    ? positional[index]
                    : fallback;
            }

            /**
             * Informs the name used to start the application.
             * @return The application name used.
             */
            inline const std::string& getAppname() const
            {
                return appname;
            }

            /**
             * Informs the number of positional arguments.
             * @return The number of positional arguments.
             */
            inline size_t getCount() const
            {
                return positional.size();
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
            inline bool has(const std::string& argname)
            {
                return args.find(argname) != args.end();
            }

            void init(const std::vector<Option>&);
            void parse(int, char **);

        protected:
            const Option& find(const std::string&) const;
    };

    extern Parser parser;
    extern const std::vector<Option> config;

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
     * Retrieves the value received by a named argument.
     * @param argname The name of the requested argument.
     * @param fallback The value to be returned if none is found.
     * @return The value of requested argument.
     */
    inline const std::string& get(const std::string& argname, const std::string& fallback = "")
    {
        return parser.get(argname, fallback);
    }

    /**
     * Retrieves the value of a positional argument.
     * @param index The requested argument index.
     * @param fallback The value to be returned if none is found.
     * @return The value of requested argument.
     */
    inline const std::string& get(uint16_t index, const std::string& fallback = "")
    {
        return parser.get(index, fallback);
    }

    /**
     * Initializes the command line arguments.
     * @param argc Number of arguments sent by command line.
     * @param argv The arguments sent by command line.
     */
    inline void init(int argc, char **argv)
    {
        parser.init(config);
        parser.parse(argc, argv);
    }
};

#endif