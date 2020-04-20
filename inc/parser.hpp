/**
 * Multiple Sequence Alignment parser header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <vector>

#include <utils.hpp>
#include <database.hpp>

namespace msa
{
    namespace parser
    {
        /**
         * A parser is a function resposible for reading a file and converting all
         * of its data to sequences added into a database.
         * @since 0.1.1
         */
        using functor = msa::functor<auto(const std::string&) -> database>;

        extern auto parse(const std::string&, const std::string& = {}) -> database;
        extern auto parse(const std::vector<std::string>&, const std::string& = {}) -> database;
    }
}
