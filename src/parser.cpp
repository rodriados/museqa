/** 
 * Multiple Sequence Alignment parser file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#include <map>
#include <string>
#include <vector>

#include <msa.hpp>
#include <parser.hpp>
#include <database.hpp>
#include <exception.hpp>

#include <parser/fasta.hpp>

/*
 * Keeps the list of available parsers and their respective
 * file extensions correspondence.
 */
static const std::map<std::string, parser::functor> dispatcher = {
    {"fasta", parser::fasta}
};

/**
 * Parses a file producing new sequences, after choosing correct parser.
 * @param filename The file to be parsed.
 * @param ext The file type to parse as.
 * @return All parsed database entries from file.
 */
std::vector<database_entry> parser::parse(const std::string& filename, const std::string& ext)
{
    const std::string extension = ext.size() ? ext : filename.substr(filename.find_last_of('.') + 1);
    const auto& pair = dispatcher.find(extension);

    enforce(pair != dispatcher.end(), "unknown parser for extension  <bold>%s</>", extension);
    watchdog::info("parsing sequence file <bold>%s</>", filename);

    return pair->second(filename);
}

/**
 * Parses a patch of files, each one according to their respective parsers.
 * @param files The list of files to parse.
 * @param ext The file type to parse as.
 * @return All parsed database entries from all files.
 */
std::vector<database_entry> parser::parse_many(const std::vector<std::string>& files, const std::string& ext)
{
    std::vector<database_entry> result;

    for(const std::string& filename : files) {
        std::vector<database_entry> fcontents = parse(filename, ext);
        result.insert(result.end(), fcontents.begin(), fcontents.end());
    }

    return result;
}
