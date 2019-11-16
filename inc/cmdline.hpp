/** 
 * Multiple Sequence Alignment command line header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef CMDLINE_HPP_INCLUDED
#define CMDLINE_HPP_INCLUDED

#include <unordered_map>
#include <string>
#include <vector>
#include <utility>

#include <exception.hpp>

namespace cmdline
{
    /**
     * An option definition which essentially represents what an option is.
     * @since 0.1.1
     */
    struct option
    {
        std::string name;                   /// The option's name, used for retrieving its value.
        std::vector<std::string> flags;     /// The strings to match to correspond to option.
        std::string description;            /// The option's description.
        bool variadic = false;              /// Does the option require any arguments?
    };

    /**
     * Parses the parameters given from the command line, and stores them so it can
     * be easily retrieved when needed.
     * @since 0.1.1
     */
    struct parser
    {
        using key_type = std::string;                       /// The parser's options key type.
        std::unordered_map<key_type, option> config;        /// The map of options from their flags.
        std::unordered_map<key_type, std::string> result;   /// The map of parsed option results.
        std::vector<std::string> positional;                /// The list of positional values.
    };
}

namespace internal
{
    namespace cmdline
    {
        /**
         * Converts the given argument to a general type.
         * @tparam T The target type to convert to.
         * @param arg The argument value to be converted.
         * @return The converted value from given argument.
         */
        template <typename T>
        inline auto convert(const std::string& arg)
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
        inline auto convert(const std::string& arg)
        -> typename std::enable_if<std::is_integral<T>::value, T>::type
        try {
            return static_cast<T>(std::stoull(arg));
        }

        catch(const std::invalid_argument&) {
            throw exception {"unable to convert argument to integer '%s'", arg};
        }

        catch(const std::out_of_range&) {
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
        inline auto convert(const std::string& arg)
        -> typename std::enable_if<std::is_floating_point<T>::value, T>::type
        try {
            return static_cast<T>(std::stold(arg));
        }

        catch(const std::invalid_argument&) {
            throw exception {"unable to convert argument to floating point '%s'", arg};
        }

        catch(const std::out_of_range&) {
            throw exception {"argument numeric value out of range '%s'", arg};
        }
    }
}

namespace cmdline
{
    /**
     * The global command line default parser instance.
     * @since 0.1.1
     */
    extern parser instance;

    extern auto prepare(const std::vector<option>&) noexcept -> std::unordered_map<std::string, option>;
    extern auto parse(int, char **) -> void;
    /**
     * Initializes the command line available options.
     * @param options The list of available options.
     */
    inline void init(const std::vector<option>& options) noexcept
    {
        instance.config = prepare(options);
    }

    /**
     * Informs the number of positional arguments.
     * @return The total number of positional arguments.
     */
    inline size_t count() noexcept
    {
        return instance.positional.size();
    }

    /**
     * Checks whether an argument exists.
     * @param name The name of the requested argument.
     * @return Does the argument exist?
     */
    inline bool has(const std::string& name) noexcept
    {
        const auto& it = instance.result.find(name);
        return it != instance.result.end() && !it->second.empty();
    }

    /**
     * Retrieves the value received by a named argument.
     * @tparam T The type the argument must be converted to.
     * @param name The name of the requested argument.
     * @param fallback The value to be returned if none is found.
     * @return The value of requested argument.
     */
    template <typename T = std::string>
    inline T get(const std::string& name, const T& fallback = {})
    {
        return has(name) ? internal::cmdline::convert<T>(instance.result[name]) : fallback;
    }

    /**
     * Retrieves the value received by a positional argument.
     * @tparam T The type the argument must be converted to.
     * @param id The id of the requested argument.
     * @param fallback The value to be returned if none is found.
     * @return The value of requested argument.
     */
    template <typename T = std::string>
    inline T get(ptrdiff_t id, const T& fallback = {})
    {
        return size_t(id) < count() ? internal::cmdline::convert<T>(instance.positional[id]) : fallback;
    }
}

#endif