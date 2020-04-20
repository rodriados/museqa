/**
 * Multiple Sequence Alignment fasta parser header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#include <string>

#include <parser.hpp>
#include <database.hpp>

namespace msa
{
    namespace parser
    {
        extern database fasta(const std::string&);
    }
}