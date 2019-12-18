/**
 * Multiple Sequence Alignment fasta parser file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <string>
#include <vector>
#include <fstream>

#include <database.hpp>
#include <exception.hpp>

#include <parser/fasta.hpp>

namespace msa
{
    /**
     * Extracts a sequence out of the file and puts it into a database.
     * @param file The file to read sequence from.
     * @param db The destination database for the sequence.
     * @return Could a sequence be extracted?
     */
    static bool extract_to(std::fstream& file, database& db)
    {
        std::string line, seq;

        while(line.size() < 1 || line[0] != 0x3E) {
            // Ignore all characters in file until a '>' is seen. This indicates
            // the beginning of a sequence description, and our sequences will always
            // have a description.
            if(file.eof())
                return false;

            std::getline(file, line);
        }

        std::string description = line.substr(1);

        while(file.peek() != 0x3E && std::getline(file, line) && line.size() > 0)
            seq.append(line);

        db.add(description, seq);
        return true;
    }

    /**
     * Reads a file and parses all sequences contained in it.
     * @param filename The name of the file to be loaded.
     * @return The sequences parsed from file.
     */
    database parser::fasta(const std::string& filename)
    {
        std::fstream file (filename, std::fstream::in);
        database result;

        enforce(!file.fail(), "file does not exist or cannot be read '%s'", filename);

        while(!file.eof() && !file.fail())
            extract_to(file, result);

        file.close();
        return result;
    }
}