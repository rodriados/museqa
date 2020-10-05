/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file String formatting and printing functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019-present Rodrigo Siqueira
 */
#pragma once

#include <cstdio>
#include <string>
#include <utility>

#include "utils.hpp"
#include "pointer.hpp"

#if defined(__museqa_compiler_gcc)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wformat-security"
#endif

namespace museqa
{
    namespace detail
    {
        namespace fmt
        {
            /**
             * Alias type for a primitive format buffer.
             * @since 0.1.1
             */
            using fmtbuffer = museqa::pointer<char[]>;

            /**
             * Checks whether all argument types are actually printable.
             * @tparam T The list of argment types to be checked.
             * @since 0.1.1
             */
            template <typename ...T>
            using valid = std::integral_constant<bool, utils::all(
                    std::is_scalar<typename std::decay<T>::type>::value...
                )>;

            /**#@+
             * Creates a string based on the given format template.
             * @param fmtstr The format template to base new string on.
             * @param args The format template arguments.
             * @return The final formatted string.
             */
            template <typename ...T>
            inline const fmtbuffer format(const char *fmtstr, T&&... args) noexcept
            {
                static_assert(valid<decltype(args)...>::value, "formatter must return scalar");
                size_t length = snprintf(nullptr, 0, fmtstr, args...) + 1;
                auto result = fmtbuffer::make(length);
                snprintf(&result, length, fmtstr, args...);
                return result;
            }

            template <typename ...T>
            inline const fmtbuffer format(const std::string& fmtstr, T&&... args) noexcept
            {
                return format(fmtstr.c_str(), args...);
            }
            /**#@-*/
        }
    }

    namespace fmt
    {
        /**
         * Allows a type to have a pre-formatting action, that is, it has the possibility
         * of formatting itself before being printed.
         * @tparam T The type being currently formatted.
         * @since 0.1.1
         */
        template <typename T>
        struct formatter
        {
            using return_type = const T&;       /// The raw parsing return type.

            /**
             * Assumes the type is scalar and simply return it.
             * @param arg The value to format.
             * @return The formatted value.
             */
            static inline auto parse(const T& arg) noexcept -> return_type
            {
                return arg;
            }
        };

        /**
         * Allows a string to be directly sent to format functions.
         * @since 0.1.1
         */
        template <>
        struct formatter<std::string>
        {
            using return_type = const char *;   /// The raw parsing return type.

            /**
             * Returns the string's internal pointer.
             * @param arg The string to format.
             * @return The string's internal pointer.
             */
            static inline auto parse(const std::string& arg) noexcept -> return_type
            {
                return arg.c_str();
            }
        };

        /**
         * Adapts a formatter into a new formatter. This allows formatters to be
         * used as helpers for new formatters to build upon.
         * @tparam T The type of which formatter must be adapted.
         * @since 0.1.1
         */
        template <typename T>
        struct adapter : public formatter<T>
        {
            T temporary;                                    /// Temporary value to be formatted.

            using base = formatter<T>;                      /// The base formatter.
            using return_type = typename base::return_type; /// The raw format return type.

            /**
             * Calls the base formatter and returns the formatted value.
             * @param value The value to be formatted.
             * @return The formatted value.
             */
            inline auto adapt(const T& value) -> return_type
            {
                return this->parse(temporary = value);
            }
        };

        /**
         * The alias for global formatters.
         * @tparam T The type to be formatted.
         * @since 0.1.1
         */
        template <typename T>
        using formatter_g = formatter<
                typename std::remove_cv<typename std::remove_reference<T>::type
            >::type>;

        /**
         * Creates a string with the given format template and arguments.
         * @tparam F The format template type.
         * @tparam T The format template arguments' types.
         * @param fmtstr The format template string.
         * @param args The format arguments.
         * @return The newly created string from format template.
         */
        template <typename F, typename ...T>
        inline std::string format(const F& fmtstr, T&&... args) noexcept
        {
            return &detail::fmt::format(fmtstr, formatter_g<T>::parse(args)...);
        }

        /**
         * Sends the formatted string directly to the given stream.
         * @tparam F The format template type.
         * @tparam T The format template arguments' types.
         * @param file The file to output the created string.
         * @param fmtstr The format template string.
         * @param args The format arguments.
         */
        template <typename F, typename ...T>
        inline void fprint(FILE *file, const F& fmtstr, T&&... args) noexcept
        {
            fputs(&detail::fmt::format(fmtstr, formatter_g<T>::parse(args)...), file);
        }

        /**
         * Sends the formatted string directly to the output.
         * @tparam F The format template type.
         * @tparam T The format template arguments' types.
         * @param color The color to apply to the whole string.
         * @param fmtstr The format template string.
         * @param args The format arguments.
         */
        template <typename F, typename ...T>
        inline void print(const F& fmtstr, T&&... args) noexcept
        {
            fprint(stdout, fmtstr, args...);
        }
    }
}

#if defined(__museqa_compiler_gcc)
  #pragma GCC diagnostic pop
#endif
