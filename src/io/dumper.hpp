/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements the base for IO dumpers of data structures into files.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#pragma once

#include <string>

#include "utils.hpp"
#include "functor.hpp"

namespace museqa
{
    namespace io
    {
        namespace base
        {
            /**
             * The common base for a given type's file dumper.
             * @tparam T The object type to be dumped into a file.
             * @since 0.1.1
             */
            template <typename T>
            struct dumper
            {
                using functor = museqa::functor<auto(const T&, const std::string&) -> bool>;

                inline dumper() noexcept = default;
                inline dumper(const dumper&) noexcept = default;
                inline dumper(dumper&&) noexcept = default;

                inline virtual ~dumper() noexcept = default;

                inline dumper& operator=(const dumper&) noexcept = default;
                inline dumper& operator=(dumper&&) noexcept = default;

                /**
                 * Dumps the given object into a file of given name.
                 * @param obj The object to be dumped into a file.
                 * @param fname The name of file to dump object into.
                 * @param ext The file extension or dumper name to be used.
                 * @return Has the object been successfully dumped?
                 */
                inline bool dump(const T& obj, const std::string& fname, const std::string& ext = {}) const
                {
                    auto type = ext.size() ? ext : utils::extension(fname);
                    auto func = factory(type);
                    return func (obj, fname);
                }

                virtual auto factory(const std::string&) const -> functor = 0;
                virtual auto list() const noexcept -> const std::vector<std::string>& = 0;
            };
        }

        /**
         * Defines the generic object dumper type. This struct must be specialized
         * to a new type, whenever a new dumpable type is introduced.
         * @tparam T The target object type to dump.
         * @since 0.1.1
         */
        template <typename T>
        struct dumper : public base::dumper<T>
        {};
    }
}
