/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements an loader for sequences database.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <vector>

#include <database.hpp>
#include "io/loader.hpp"

namespace museqa
{
    namespace io
    {
        /**
         * Specializes a loader for our sequence database type.
         * @since 0.1.1
         */
        template <>
        struct loader<database> : public base::loader<database>
        {
            auto factory(const std::string&) const -> functor override;
            auto validate(const std::string&) const noexcept -> bool override;
            auto list() const noexcept -> const std::vector<std::string>& override;
        };

        namespace parser
        {
            /*
             * Declaration of all available parsers to the target datatype. 
             */
            extern auto fasta(const std::string&) -> database;
        }
    }
}
