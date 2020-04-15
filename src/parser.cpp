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

namespace msa
{
    /*
     * Keeps the list of available parsers and their respective
     * file extensions correspondence.
     */
    static const std::map<std::string, parser::functor> dispatcher = {
        {"fa", parser::fasta}
    ,   {"fasta", parser::fasta}
    };

    /**
     * Parses a file producing new sequences, after choosing correct parser.
     * @param filename The file to be parsed.
     * @param ext The file type to parse as.
     * @return All parsed database entries from file.
     */
    database parser::parse(const std::string& filename, const std::string& ext)
    {
        const std::string extension = ext.size() ? ext : filename.substr(filename.find_last_of('.') + 1);
        const auto& pair = dispatcher.find(extension);

        enforce(pair != dispatcher.end(), "unknown parser for extension '%s'", extension);
        watchdog::info("parsing sequence file '<bold>%s</>'", filename);

        return pair->second(filename);
    }

    /**
     * Parses a patch of files, each one according to their respective parsers.
     * @param files The list of files to parse.
     * @param ext The file type to parse as.
     * @return All parsed database entries from all files.
     */
    database parser::parse(const std::vector<std::string>& files, const std::string& ext)
    {
        database result;

        for(const std::string& filename : files)
            result.merge(parse(filename, ext));
        
        return result;
    }
}