/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implementation for the FASTA parser of sequences database.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#include <string>
#include <vector>
#include <fstream>

#include "database.hpp"
#include "exception.hpp"

#include "io/loader/database.hpp"

using namespace museqa;

namespace
{
    /**
     * Extracts a sequence out of the file and puts it into a database.
     * @param file The file to read sequence from.
     * @param db The destination database for the sequence.
     * @return Could a sequence be extracted?
     */
    static bool extract_to_db(std::fstream& file, database& db)
    {
        std::string line, contents;
        std::string description;

        while(line.size() < 1 || line[0] != 0x3E) {
            // We must ignore all characters on file until a ">" is seen. This
            // symbol indicates the beginning of a sequence description, and
            // in this file format, sequences must always have a description.
            if(file.eof()) return false;
            std::getline(file, line);
        }

        description = line.substr(1);

        while(file.peek() != 0x3E && std::getline(file, line) && line.size() > 0)
            contents.append(line);

        db.add(description, contents);
        return true;
    }
}

namespace museqa
{
    namespace io
    {
        /**
         * Reads a file and parses all sequences contained in it.
         * @param filename The name of the file to be loaded.
         * @return The sequences parsed from file.
         */
        auto parser::fasta(const std::string& filename) -> database
        {
            database result;
            std::fstream file (filename, std::fstream::in);

            enforce(!file.fail(), "file does not exist or cannot be read '%s'", filename);

            while(!file.eof() && !file.fail())
                extract_to_db(file, result);

            file.close();
            return result;
        }
    }
}
