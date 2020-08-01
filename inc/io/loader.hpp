/**
 * Multiple Sequence Alignment IO object loader header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020 Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <vector>
#include <utility>

#include <utils.hpp>

namespace msa
{
    namespace detail
    {
        namespace io
        {
            /**
             * The common base for a given type's loader.
             * @tparam T The target object type to be loaded from file.
             * @since 0.1.1
             */
            template <typename T>
            struct loader
            {
                using functor = msa::functor<auto(const std::string&) -> T>;

                inline loader() noexcept = default;
                inline loader(const loader&) noexcept = default;
                inline loader(loader&&) noexcept = default;

                inline virtual ~loader() noexcept = default;

                inline loader& operator=(const loader&) noexcept = default;
                inline loader& operator=(loader&&) noexcept = default;

                /**
                 * Loads the given file content's to an object instance.
                 * @param fname The target file name to be loaded.
                 * @param ext The file extension or parser name to be used.
                 * @return The newly instantiate object loaded from file.
                 */
                inline T load(const std::string& fname, const std::string& ext = {}) const
                {
                    auto fext = ext.size() ? ext : fname.substr(fname.find_last_of('.') + 1);
                    auto func = factory(fext);
                    return func (fname);
                }

                virtual auto factory(const std::string&) const -> functor = 0;
                virtual auto validate(const std::string&) const noexcept -> bool = 0;
                virtual auto list() const noexcept -> const std::vector<std::string>& = 0;
            };
        }
    }

    namespace io
    {
        /**
         * The common base for all object loaders.
         * @tparam T The type of object to be loaded from file.
         * @since 0.1.1
         */
        template <typename T>
        using base_loader = detail::io::loader<T>;

        /**
         * Defines the generic object loader type. Whenever a new type may loaded
         * directly from a file, this struct must be specialized for given type.
         * @tparam T The target object type to load.
         * @since 0.1.1
         */
        template <typename T>
        struct loader : public base_loader<T>
        {};
    }
}
