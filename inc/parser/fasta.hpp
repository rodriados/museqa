/**
 * Multiple Sequence Alignment fasta parser header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PARSER_FASTA_HPP_INCLUDED
#define PARSER_FASTA_HPP_INCLUDED

#include <string>
#include <vector>

#include "parser.hpp"
#include "database.hpp"

namespace parser
{
    extern std::vector<DatabaseEntry> fasta(const std::string&);
};

#endif