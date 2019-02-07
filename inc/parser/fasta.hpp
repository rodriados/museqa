/**
 * Multiple Sequence Alignment fasta parser header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PARSER_FASTA_HPP_INCLUDED
#define PARSER_FASTA_HPP_INCLUDED

#include <fstream>

#include "parser.hpp"
#include "database.hpp"

namespace parser
{
    extern bool fasta(std::fstream&, DatabaseEntry&);
};

#endif