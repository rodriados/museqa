/** 
 * Multiple Sequence Alignment parser file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#include <map>
#include <string>
#include <vector>

#include "parser.hpp"
#include "database.hpp"
#include "exception.hpp"

#include "parser/fasta.hpp"

/*
 * Keeps the list of available parsers and their respective
 * file extensions correspondence.
 */
static const std::map<std::string, parser::Parser> dispatcher = {
    {"fasta", parser::fasta}
};

/**
 * Parses a file producing new sequences, after choosing correct parser.
 * @param filename The file to be parsed.
 * @param ext The file type to parse as.
 * @return All parsed database entries from file.
 */
std::vector<DatabaseEntry> parser::parse(const std::string& filename, const std::string& ext)
{
    std::string extension = ext.size() ? ext : filename.substr(filename.find_last_of('.') + 1);
    const auto& pair = dispatcher.find(extension);

    if(pair == dispatcher.end())
        throw Exception("unknown parser for extension: " + extension);

    return pair->second(filename);
}

/**
 * Parses a patch of files, each one according to their respective parsers.
 * @param files The list of files to parse.
 * @param ext The file type to parse as.
 * @return All parsed database entries from all files.
 */
std::vector<DatabaseEntry> parser::parseMany(const std::vector<std::string>& files, const std::string& ext)
{
    std::vector<DatabaseEntry> result;

    for(const std::string& filename : files) {
        std::vector<DatabaseEntry> sequences = parse(filename, ext);
        result.insert(result.end(), sequences.begin(), sequences.end());
    }

    return result;
}
