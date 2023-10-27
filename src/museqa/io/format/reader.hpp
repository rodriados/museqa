/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The common base for all format file readers.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2023-present Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <fstream>
#include <utility>

#include <museqa/guard.hpp>
#include <museqa/environment.h>
#include <museqa/io/exception.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace io::format
{
    /**
     * The abstract base type for a format reader. Concrete reader implementations
     * may provide specific and more specialized methods for reading target type.
     * @tparam T The target type for reader.
     * @since 1.0
     */
    template <typename T>
    class reader_t
    {
        protected:
            std::ifstream m_fstream;

        public:
            inline reader_t() noexcept = default;
            inline reader_t(const reader_t&) noexcept = delete;
            inline reader_t(reader_t&&) noexcept = default;

            /**
             * Initializes a new reader from a file name.
             * @param fname The name of file to read from.
             */
            inline explicit reader_t(const std::string& fname)
              : m_fstream (fname, std::ios::in)
            {
                museqa::guard<io::exception_t>(
                    !m_fstream.fail()
                  , "file does not exist or is not readable"
                );
            }

            /**
             * Initializes a new reader by acquiring ownership of a file stream.
             * @param fstream The file stream to read from.
             */
            inline explicit reader_t(std::ifstream&& fstream)
              : m_fstream (std::forward<decltype(fstream)>(fstream))
            {}

            inline reader_t& operator=(const reader_t&) noexcept = delete;
            inline reader_t& operator=(reader_t&&) noexcept = default;

            inline virtual ~reader_t() = default;

            /**
             * Reads an instance of the target type and returns it.
             * @return The target type instance read from file.
             */
            virtual auto read() -> T = 0;
    };
}

MUSEQA_END_NAMESPACE
