/** 
 * Multiple Sequence Alignment command line header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef CLI_HPP_INCLUDED
#define CLI_HPP_INCLUDED

#pragma once

#include <string>
#include <vector>
#include <map>

/**
 * Stores all information about a given option available from the command
 * line. There should be an instance for each option available.
 * @since 0.1.alpha
 */
class Option
{
    protected:
        std::string sname;            /// The option's short name.
        std::string lname;            /// The option's long name.
        std::string description;      /// The option's description.
        std::string argument;         /// The option's argument name if any

    public:
        Option() noexcept = default;

        /**
         * Builds an option from its names and description.
         * @param sname The option's short name.
         * @param lname The option's long name.
         * @param description The option's description.
         * @param argument The option's argument name if any.
         */
        inline Option
            (   const std::string& sname
            ,   const std::string& lname
            ,   const std::string& description
            ,   const std::string& argument = ""    )
        :   sname(sname)
        ,   lname(lname)
        ,   description(description)
        ,   argument(argument) {}

        /**
         * The copy operator. This operator is needed so we can initialize our options
         * outside the class constructors.
         * @param other The option instance to be copied.
         * @return This instance for method chaining.
         */
        inline Option& operator=(const Option& other)
        {
            this->sname = other.sname;
            this->lname = other.lname;
            this->description = other.description;
            this->argument = other.argument;
            return *this;
        }

        /**
         * Gets the option's short name.
         * @return The retrieved option short name.
         */
        inline const std::string& getSname() const
        {
            return this->sname;
        }

        /**
         * Gets the option's long name.
         * @return The retrieved option long name.
         */
        inline const std::string& getLname() const
        {
            return this->lname;
        }

        /**
         * Gets the option's description.
         * @return The retrieved option description.
         */
        inline const std::string& getDescription() const
        {
            return this->description;
        }

        /**
         * Gets the option's argument name.
         * @return The retrieved option argument name.
         */
        inline const std::string& getArgument() const
        {
            return this->argument;
        }

        /**
         * Checks whether the option is empty or unknown.
         * @return Is the option unknown?
         */
        inline bool isUnknown() const
        {
            return this->sname.empty()
                && this->lname.empty();
        }

        /**
         * Checks whether the option requires an argument value or not.
         * @return Does the option require an argument?
         */
        inline bool isVariadic() const
        {
            return !this->argument.empty();
        }
};

/**
 * Parses the parameters given from the command line, and stores them so it can
 * be easily retrieved when needed.
 * @since 0.1.alpha
 */
class CliParser
{
    protected:
        std::string appname;                        /// The name used by the application.
        std::vector<Option> options;                /// The list of options available.
        std::vector<std::string> arguments;         /// The list of required (and positional) arguments.
        std::map<std::string, std::string> values;  /// The list of parsed values.

    public:
        CliParser() noexcept = default;

        /**
         * Checks whether an argument exists.
         * @param argname The name of the requested argument.
         * @return Does the argument exist?
         */
        inline bool has(const std::string& argname) const
        {
            return this->values.find(argname) != this->values.end();
        }

        /**
         * Retrieves the value received by a named argument.
         * @param argname The name of the requested argument.
         * @param fallback The value to be returned if none is found.
         * @return The value of requested argument.
         */
        inline const std::string& get(const std::string& argname, const std::string& fallback = "") const
        {
            return this->has(argname)
                ? this->values.find(argname)->second
                : fallback;
        }

        /**
         * Informs the name used to start the application.
         * @return The application name used.
         */
        inline const std::string& getAppname() const
        {
            return this->appname;
        }

        /**
         * Gives access to the options given to the input module.
         * @return The application options.
         */
        inline const std::vector<Option>& getOptions() const
        {
            return this->options;
        }

        void init(const std::vector<Option>&, const std::vector<std::string>&);
        void parse(int, char **);

    private:
        const Option& find(const std::string&) const;
};

extern CliParser cli;

#endif