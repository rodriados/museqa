/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The implementation of FASTA format file reader.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2023-present Rodrigo Siqueira
 */
#include <map>
#include <string>
#include <fstream>

#include <museqa/environment.h>
#include <museqa/bio/sequence.hpp>
#include <museqa/io/format/fasta/reader.hpp>

MUSEQA_BEGIN_NAMESPACE

inline static bool is_stream_healthy(std::ifstream&);
inline static constexpr bool is_token_comment(int);
inline static constexpr bool is_token_description(int);

/**
 * Extracts a sequence from a file stream of FASTA format.
 * @param fstream The file stream to extract a sequence from.
 * @return The extracted sequence data.
 */
static auto extract_sequence_from_stream(std::ifstream& fstream) -> bio::sequence::data_t
{
    std::string line, contents;

    while (line.size() < 1 || !is_token_description(line[0])) {
        // We must skip and ignore all lines on file until we detect one that starts
        // with greater-than or semicolon. These symbols indicate a line for sequence
        // description, which in this file format must always be present.
        if (!is_stream_healthy(fstream))
            return bio::sequence::data_t();
        std::getline(fstream, line);
    }

    std::string description = line.substr(1);

    // If another line starts with a semicolon, than it should be considered a comment
    // and therefore ignored. Comments are out-dated and should be avoided, but
    // we must support them anyway. Differently from the original file format description,
    // we do accept lines started with semicolon as comments even though the sequence
    // description itself might have been represented with a greater-than symbol.
    while (is_stream_healthy(fstream) && is_token_comment(fstream.peek()))
        std::getline(fstream, line);

    // Now that the description has been read and all possible comments have been
    // ignored, we must read the sequence simply by concatenating every line until
    // a new sequence or an empty line is detected.
    while (is_stream_healthy(fstream) && !is_token_description(fstream.peek()))
        if (std::getline(fstream, line); line.size() > 0)
            contents.append(line);
        else break;

    return bio::sequence::data_t {
        bio::sequence::encode(contents)
      , std::map<bio::sequence::attribute_t, std::string> {
            {bio::sequence::attribute_t::description, description}
        }
    };
}

/**
 * Checks whether the file stream is healthy and can be read from.
 * @param fstream The file stream to check if healthy.
 * @return Is the stream healthy?
 */
inline static bool is_stream_healthy(std::ifstream& fstream)
{
    return !fstream.eof() && !fstream.fail();
}

/**
 * Checks whether the given token indicates the representation of a comment.
 * @param token The token to be checked for a comment indication.
 * @return Does the token indicate a comment is next to come?
 */
inline static constexpr bool is_token_comment(int token)
{
    return token == ';';
}

/**
 * Checks whether the given token indicates the representation of a sequence description.
 * @param token The token to be checked for a sequence description indication.
 * @return Does the token indicate a sequence description is next to come?
 */
inline static constexpr bool is_token_description(int token)
{
    return token == '>' || is_token_comment(token);
}

/**
 * Reads a sequence from the FASTA file.
 * @return The sequence extracted from file.
 */
auto io::format::fasta::reader_t::read() -> bio::sequence::data_t
{
    return extract_sequence_from_stream(this->m_fstream);
}

MUSEQA_END_NAMESPACE
