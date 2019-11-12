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
#include <fstream>

#include <utils.hpp>
#include <database.hpp>

namespace parser
{
    /**
     * A parser is a function resposible for reading a file and converting
     * its data to sequences that can be added to the database.
     * @since 0.1.1
     */
    using functor = ::functor<std::vector<database_entry>(const std::string&)>;

    extern std::vector<database_entry> parse(const std::string&, const std::string& = {});
    extern std::vector<database_entry> parse_many(const std::vector<std::string>&, const std::string& = {});
};

#endif