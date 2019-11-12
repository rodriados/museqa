/** 
 * Multiple Sequence Alignment command line header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef CMDLINE_HPP_INCLUDED
#define CMDLINE_HPP_INCLUDED

#include <string>
#include <vector>
#include <cstdlib>
#include <utility>
#include <map>

namespace cmdline
{
    /**
     * Stores all information about a given option available from the command
     * line. There should be an instance for each option available.
     * @since 0.1.1
     */
    class option
    {
        protected:
            std::string mshort;     /// The option's short name alternative.
            std::string mlong;      /// The option's long name alternative.
            bool mvariadic;         /// Is the option variadic, so it requires a value?
            bool mrequired;         /// Is the option required for software execution?

        public:
            inline option() = default;
            inline option(const option&) = default;
            inline option(option&&) noexcept = default;

            /**
             * Builds a new option instance.
             * @param nshort The option's short name.
             * @param nlong The option's long name.
             * @param (ignored) The option's description.
             * @param variadic Does the option require a value?
             * @param required Is the option required?
             */
            inline option(
                    const char *nshort
                ,   const char *nlong
                ,   const char * //description
                ,   const bool variadic = false
                ,   const bool required = false
                )
                noexcept
            :   mshort {nshort}
            ,   mlong {nlong}
            ,   mvariadic {variadic}
            ,   mrequired {required}
            {}

            inline option& operator=(const option&) = default;
            inline option& operator=(option&&) = default;

            /**
             * Checks whether the a string correspond to this option.
             * @param tgt The string to check and match with.
             * @return Is this the requested option?
             */
            inline bool operator==(const std::string& tgt) const noexcept
            {
                return tgt == mshort || tgt == mlong;
            }

            /**
             * Checks whether the option is empty or unknown.
             * @return Is the option unknown?
             */
            inline bool empty() const noexcept
            {
                return mshort.empty() && mlong.empty();
            }

            /**
             * Checks whether the option is required.
             * @return Is the option required?
             */
            inline bool required() const noexcept
            {
                return mrequired;
            }

            /**
             * Checks whether the option is variadic.
             * @return Is the option variadic.
             */
            inline bool variadic() const noexcept
            {
                return mvariadic;
            }

            /**
             * Gets the option's short name.
             * @return The retrieved option short name.
             */
            inline const std::string& shortname() const noexcept
            {
                return mshort;
            }

            /**
             * Gets the option's long name.
             * @return The retrieved option long name.
             */
            inline const std::string& longname() const noexcept
            {
                return mlong;
            }
    };

    /**
     * Parses the parameters given from the command line, and stores them so it can
     * be easily retrieved when needed.
     * @since 0.1.1
     */
    class parser
    {
        private:
            std::vector<std::string> mrequired;             /// The list of required options.
            std::map<std::string, option> moptions;         /// The map of available options.

        protected:
            std::string mappname;                           /// The application's name.
            std::vector<std::string> mpositional;           /// The list of positional values.
            std::map<std::string, std::string> mvalues;     /// The map of parsed option values.

        public:
            inline parser() noexcept = default;
            inline parser(const parser&) = default;
            inline parser(parser&&) = default;

            parser(const std::vector<option>&) noexcept;

            inline parser& operator=(const parser&) = default;
            inline parser& operator=(parser&&) = default;

            virtual ~parser() noexcept = default;

            /**#@+
             * Retrieves the value received by a named argument.
             * @param arg The name of the requested argument.
             * @return The value of requested argument.
             */
            inline const std::string& get(const std::string& arg) const noexcept
            {
                return mvalues.find(arg)->second;
            }

            inline const std::string& get(size_t arg) const noexcept
            {
                return mpositional[arg];
            }
            /**#@-*/

            /**#@+
             * Checks whether an argument has been parsed.
             * @param arg The name of the requested argument.
             * @return Does the argument exist?
             */
            inline bool has(const std::string& arg) const noexcept
            {
                return mvalues.find(arg) != mvalues.end();
            }

            inline bool has(size_t arg) const noexcept
            {
                return arg < mpositional.size();
            }
            /**#@-*/

            /**
             * Informs the number of positional arguments.
             * @return The number of positional arguments.
             */
            inline size_t count() const noexcept
            {
                return mpositional.size();
            }

            /**
             * Informs the name used to start the application.
             * @return The application name used.
             */
            inline const std::string& appname() const noexcept
            {
                return mappname;
            }

            virtual void parse(int, char **);

        protected:
            const option& find(const std::string&) const noexcept;
    };

    /**
     * The global command line parser singleton instance.
     * @since 0.1.1
     */
    extern parser singleton;

    /**
     * Initializes the command line arguments.
     * @param options The options available.
     */
    inline void init(const std::vector<option>& options)
    {
        singleton = parser {options};
    }

    /**
     * Parses the command line arguments.
     * @param argc Number of arguments sent by command line.
     * @param argv The arguments sent by command line.
     */
    inline void parse(int argc, char **argv)
    {
        singleton.parse(argc, argv);
    }

    /**
     * Informs the number of positional arguments.
     * @return The total number of positional arguments.
     */
    inline size_t count() noexcept
    {
        return singleton.count();
    }

    /**
     * Informs the name used to start the application.
     * @return The application's name used.
     */
    inline const std::string& appname() noexcept
    {
        return singleton.appname();
    }

    /**
     * Checks whether an argument exists.
     * @param arg The name of the requested argument.
     * @return Does the argument exist?
     */
    template <typename P>
    inline bool has(const P& arg) noexcept
    {
        return singleton.has(arg);
    }

    /**#@+
     * Retrieves the value received by a named argument.
     * @tparam T The type the argument must be converted to.
     * @param arg The name of the requested argument.
     * @param fallback The value to be returned if none is found.
     * @return The value of requested argument.
     */
    template <typename T = std::string, typename P>
    inline auto get(const P& arg, const T& fallback = {}) noexcept
    -> typename std::enable_if<std::is_convertible<std::string, T>::value, T>::type
    {
        return singleton.has(arg)
            ? static_cast<T>(singleton.get(arg))
            : fallback;
    }

    template <typename T, typename P>
    inline auto get(const P& arg, const T& fallback = {}) noexcept
    -> typename std::enable_if<std::is_integral<T>::value, T>::type
    {
        return singleton.has(arg)
            ? static_cast<T>(strtoull(singleton.get(arg).c_str(), nullptr, 0))
            : fallback;
    }

    template <typename T, typename P>
    inline auto get(const P& arg, const T& fallback = {}) noexcept
    -> typename std::enable_if<std::is_floating_point<T>::value, T>::type
    {
        return singleton.has(arg)
            ? static_cast<T>(strtold(singleton.get(arg).c_str(), nullptr))
            : fallback;
    }
    /**#@-*/
}

#endif