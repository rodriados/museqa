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
 * Reads a file and parses all sequences contained in it.
 * @param filename The name of the file to be loaded.
 * @param parsef The parser function to use for parsing the file.
 * @return The sequences parsed from file.
 */
std::vector<DatabaseEntry> readfile(const std::string& filename, parser::Parser parsef)
{
    std::fstream file(filename, std::fstream::in);
    std::vector<DatabaseEntry> result;

    DatabaseEntry entry;

    if(file.fail())
        throw Exception("'" + filename + "' is not a file or does not exist");

    while(!file.eof() && !file.fail())
        if(parsef(file, entry))
            result.push_back(entry);

    file.close();

    return result;
}

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
        throw Exception("unknown file extension '" + extension + "'");

    return readfile(filename, pair->second);
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
