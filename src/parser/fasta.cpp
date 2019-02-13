/**
 * Multiple Sequence Alignment fasta parser file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <string>
#include <vector>
#include <fstream>

#include "database.hpp"
#include "exception.hpp"

#include "parser/fasta.hpp"

/**
 * Extracts a sequence out of the file and puts it into an entry.
 * @param file The file to read sequence from.
 * @param entry The destination entry for the sequence.
 * @return Could a sequence be extracted?
 */
static bool extract(std::fstream& file, DatabaseEntry& entry)
{
    std::string line, sequence;

    while(line.size() < 1 || line[0] != 0x3E) {
        // Ignore all characters until a '>' is seen.
        // Our sequences will always have a description.
        if(file.eof()) return false;
        std::getline(file, line);
    }

    entry.description = line.substr(1);

    while(file.peek() != 0x3E && std::getline(file, line) && line.size() > 0)
        sequence.append(line);

    entry.sequence = sequence;

    return true;
}

/**
 * Reads a file and parses all sequences contained in it.
 * @param filename The name of the file to be loaded.
 * @param parsef The parser function to use for parsing the file.
 * @return The sequences parsed from file.
 */
std::vector<DatabaseEntry> parser::fasta(const std::string& filename)
{
    std::fstream file(filename, std::fstream::in);
    std::vector<DatabaseEntry> result;

    DatabaseEntry entry;

    if(file.fail())
        throw Exception("'" + filename + "' is not a file or does not exist");

    while(!file.eof() && !file.fail())
        if(extract(file, entry))
            result.push_back(entry);

    file.close();

    return result;
}