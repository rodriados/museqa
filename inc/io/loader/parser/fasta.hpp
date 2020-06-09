/**
 * Multiple Sequence Alignment fasta parser header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2020 Rodrigo Siqueira
 */
#pragma once

#include <string>

#include <utils.hpp>
#include <database.hpp>

namespace msa
{
    namespace io
    {
        namespace parser
        {
            /**
             * Parses a file into a database instance.
             * @param filename The file to be parsed.
             * @return The newly parsed database instance.
             */
            extern auto fasta(const std::string& filename) -> database;
        }
    }
}
