/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The implementation of FASTA format file reader.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2023-present Rodrigo Siqueira
 */
#include <string>
#include <istream>

#include <museqa/environment.h>
#include <museqa/bio/sequence/dataset.hpp>
#include <museqa/bio/sequence/encoder.hpp>
#include <museqa/bio/sequence/attribute.hpp>
#include <museqa/io/format/fasta/reader.hpp>

#define TOKEN_COMMENT ';'
#define TOKEN_DESCRIPTION '>'

MUSEQA_BEGIN_NAMESPACE

namespace
{
    /**
     * Checks whether the stream is healthy and can be read from.
     * @param stream The stream to check if healthy.
     * @return Is the stream healthy?
     */
    MUSEQA_INLINE static bool is_stream_healthy(std::istream& stream)
    {
        return !stream.eof() && !stream.fail();
    }

    /**
     * Checks whether the given token indicates the start of a comment.
     * @param token The token to be checked for a comment indication.
     * @return Does the token indicate a line comment?
     */
    MUSEQA_INLINE static bool is_token_comment(int token)
    {
        return token == TOKEN_COMMENT;
    }

    /**
     * Checks whether the given token indicates the start of a sequence description.
     * @param token The token to be checked for a sequence description indication.
     * @return Does the token indicate a sequence description line?
     */
    MUSEQA_INLINE static bool is_token_description(int token)
    {
        return token == TOKEN_DESCRIPTION || is_token_comment(token);
    }

    /**
     * Extracts a sequence from a stream of FASTA format.
     * @param stream The stream to extract a sequence from.
     * @return The extracted sequence data.
     */
    static auto read_sequence(std::istream& stream)
    -> std::pair<std::string, bio::sequence::data_t>
    {
        std::string line, contents;

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
                contents.append(line);
            else break;

        return std::pair(description, bio::sequence::data_t {
            bio::sequence::encode(contents)
          #ifdef MUSEQA_ENABLE_SEQUENCE_ATTRIBUTES
            , bio::sequence::attribute::bag_t()
          #endif
        });
    }
}

/**
 * Extracts all sequences in FASTA format from the given stream.
 * @param stream The FASTA format stream to be parsed.
 * @return A pointer to the parsed sequence dataset.
 */
auto io::format::fasta::reader_t::read_from_stream(std::istream& stream) const
-> memory::pointer::unique_t<bio::sequence::dataset_t>
{
    auto dataset = factory::memory::pointer::unique<bio::sequence::dataset_t>();

    while (!stream.eof() && !stream.fail())
        if (auto [desc, sequence] = read_sequence(stream); !sequence.buffer.empty())
            dataset->try_emplace(desc, sequence);
        else break;

    return dataset;
}

MUSEQA_END_NAMESPACE
