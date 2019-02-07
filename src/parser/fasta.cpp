/**
 * Multiple Sequence Alignment fasta parser file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <string>
#include <fstream>

#include "database.hpp"
#include "parser/fasta.hpp"

/**
 * Extracts a sequence out of the file and puts it into an entry.
 * @param file The file to read sequence from.
 * @param entry The destination entry for the sequence.
 * @return Could a sequence be extracted?
 */
bool parser::fasta(std::fstream& file, DatabaseEntry& entry)
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
