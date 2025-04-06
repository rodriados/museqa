/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The implementation of the FASTA format file reader.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2023-present Rodrigo Siqueira
 */
#include <string>
#include <istream>

#include <museqa/environment.h>
#include <museqa/bio/sequence/dataset.hpp>
#include <museqa/bio/sequence/encoder.hpp>
#include <museqa/bio/sequence/attribute.hpp>
#include <museqa/io/format/dataset/fasta/reader.hpp>

#define MUSEQA_FASTA_TOKEN_COMMENT ';'
#define MUSEQA_FASTA_TOKEN_DESCRIPTION '>'

MUSEQA_BEGIN_NAMESPACE

namespace
{
    /**
     * Checks whether the stream is healthy and ready to be read from.
     * @param stream The stream to check if healthy and ready.
     * @return Is the stream healthy and ready?
     */
    MUSEQA_INLINE static bool is_stream_healthy(std::istream& stream)
    {
        return !stream.eof() && !stream.fail();
    }

    /**
     * Checks whether the given token indicates the start of a comment line.
     * @param token The token to be checked for a comment indication.
     * @return Does the given token indicate a comment line?
     */
    MUSEQA_CONSTEXPR static bool is_token_comment(int token)
    {
        return token == MUSEQA_FASTA_TOKEN_COMMENT;
    }

    /**
     * Checks whether the given token indicates the start of a description line.
     * @param token The token to be checked for a sequence description indication.
     * @return Does the token indicate a sequence description line?
     */
    MUSEQA_CONSTEXPR static bool is_token_description(int token)
    {
        return token == MUSEQA_FASTA_TOKEN_DESCRIPTION || is_token_comment(token);
    }

    /**
     * Extracts a sequence from a stream in FASTA format.
     * @param stream The stream to extract a sequence from.
     * @return The sequence data read from the stream.
     */
    static bio::sequence::data_t read_fasta_from_stream(std::istream& stream)
    {
        std::string line, sequence;

        while (line.empty() || !is_token_description(line[0])) {
            // We must skip and ignore all lines on stream until we detect one that
            // starts with a description token, which must always be present.
            if (!is_stream_healthy(stream))
                return {};
            std::getline(stream, line);
        }

        // If the stream is still healthy and a description token has been found,
        // then we must remove the line's first character to have a sequence description.
        std::string description = line.substr(1);

        // If another line starts with a comment, than it should be ignored. Comments
        // are outdated and should be avoided, but we must support them anyway.
        // Differently from the original file format description, we do accept lines
        // started with semicolon as comments even though the sequence description
        // itself might have been represented with a greater-than symbol.
        while (is_stream_healthy(stream) && is_token_comment(stream.peek()))
            std::getline(stream, line);

        // Now that the description has been read and all possible comments have
        // been ignored, we must read the sequence simply by concatenating every
        // line until a new sequence or an empty line is detected.
        while (is_stream_healthy(stream) && !is_token_description(stream.peek()))
            if (std::getline(stream, line); line.size() > 0)
                sequence.append(line);
            else break;

        return bio::sequence::data_t(
            bio::sequence::encode(sequence)
          , bio::sequence::attribute::bag_t { description }
        );
    }
}

/**
 * Reads a sequence dataset in FASTA format from a stream.
 * @param dataset The dataset instance to read into from stream.
 * @param stream The stream to read a sequence dataset from.
 */
void io::format::dataset::fasta::reader_t::read_from_stream(
    dataset_ptr_t& dataset
  , std::istream& stream
) const {
    while (!stream.eof() && !stream.fail()) {
        if (auto sequence = read_fasta_from_stream(stream); !sequence.buffer.empty())
            dataset->push_back(sequence);
        else break;
    }
}

MUSEQA_END_NAMESPACE
