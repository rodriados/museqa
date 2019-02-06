/**
 * Multiple Sequence Alignment parser header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PARSER_HPP_INCLUDED
#define PARSER_HPP_INCLUDED

#include <string>
#include <vector>

#include "database.hpp"

namespace parser
{
    /**
     * A parser is a function resposible for reading a file and converting
     * its data to sequences that can be added to the database.
     * @since 0.1.1
     */
    using Parser = std::vector<DatabaseEntry> (*)(const std::string&);

    extern std::vector<DatabaseEntry> parse(const std::string&);
    extern std::vector<DatabaseEntry> parseMany(const std::vector<std::string>&);
};

#endif