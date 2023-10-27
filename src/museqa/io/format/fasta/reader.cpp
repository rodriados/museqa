/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The implementation of FASTA format file reader.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2023-present Rodrigo Siqueira
 */
#include <string>
#include <fstream>

#include <museqa/environment.h>
#include <museqa/bio/sequence.hpp>
#include <museqa/io/format/fasta/reader.hpp>

MUSEQA_BEGIN_NAMESPACE

/**
 * Extracts a sequence from a file stream of FASTA format.
 * @param fstream The file stream to extract a sequence from.
 * @return The extracted sequence.
 */
static auto extract_sequence_from_stream(std::ifstream& fstream) -> bio::sequence_t
{
    if (fstream.eof() || fstream.fail())
        return bio::sequence_t();

    std::string line, contents;
    bool parsing_fasta_sequence = false;

    while (line.size() < 1 || line[0] != 0x3E) {
        // We must ignore all characters on file until a ">" is seen. This symbol
        // indicates the beginning of a sequence description, and in this file format,
        // sequences must always have a description.
        if (fstream.eof()) return bio::sequence_t();
        std::getline(fstream, line);
    }

    while (fstream.peek() != 0x3E && std::getline(fstream, line) && line.size() > 0)
        contents.append(line);

    return bio::sequence::encode(contents);
}

/**
 * Reads a sequence from the FASTA file.
 * @return The sequence extracted from file.
 */
auto io::format::fasta::reader_t::read() -> bio::sequence_t
{
    return extract_sequence_from_stream(this->m_fstream);
}

MUSEQA_END_NAMESPACE
