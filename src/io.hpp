/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Exposes an interface for the IO module.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#pragma once

#include <vector>

#include "io/io.hpp"
#include "terminal.hpp"

#include "io/loader/database.hpp"

namespace museqa
{
    namespace io
    {
        /**
         * Loads the given data structure from the given file.
         * @tparam T The data structure type to be loaded from file.
         * @param filename The file to be parsed into the requested data structure.
         * @return The data structure parsed from the file.
         */
        template <typename T>
        inline auto load(const std::string& filename) -> T
        {
            static_assert(std::is_constructible<loader<T>>::value, "cannot instantiate type loader");
            static_assert(std::is_base_of<base::loader<T>, loader<T>>::value, "invalid loader type");

            auto loader = io::loader<T> {};
            return loader.load(filename);
        }

        /**
         * Dumps the given object into a file.
         * @tparam T The type of object to be dumped.
         * @param obj The object to be dumped into a file.
         * @param filename The name of the file to dumpt object into.
         * @return Has the object been successfully dumped?
         */
        template <typename T>
        inline auto dump(const T& obj, const std::string& filename) -> bool
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
        inline auto make(const std::vector<terminal::option>& options, int argc, char **argv) -> manager
        {
            return manager::make(options, argc, argv);
        }
    }
}
