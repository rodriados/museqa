/**
 * Multiple Sequence Alignment database loader header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020 Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <vector>

#include <database.hpp>
#include <io/loader.hpp>

namespace msa
{
    namespace io
    {
        /**
         * Specializes a loader for our sequence database type.
         * @since 0.1.1
         */
        template <>
        struct loader<database> : public base_loader<database>
        {
            auto factory(const std::string&) const -> functor override;
            auto validate(const std::string&) const noexcept -> bool override;
            auto list() const noexcept -> const std::vector<std::string>& override;
        };
    }
}
