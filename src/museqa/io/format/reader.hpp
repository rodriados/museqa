/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The common base for all format file readers.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2023-present Rodrigo Siqueira
 */
#pragma once

#include <filesystem>

#include <museqa/environment.h>
#include <museqa/memory/pointer/unique.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace io::format
{
    /**
     * The abstract base type for a format reader. Concrete reader implementations
     * may provide specific and more specialized methods for reading target type.
     * @tparam T The expected type to be produced from reading the file.
     * @since 1.0
     */
    template <typename T>
    struct reader_t
    {
        typedef T target_t;

        MUSEQA_CONSTEXPR reader_t() = default;
        MUSEQA_CONSTEXPR reader_t(const reader_t&) = default;
        MUSEQA_CONSTEXPR reader_t(reader_t&&) = default;

        MUSEQA_INLINE reader_t& operator=(const reader_t&) = default;
        MUSEQA_INLINE reader_t& operator=(reader_t&&) = default;

        MUSEQA_CONSTEXPR virtual ~reader_t() = default;

        /**
         * Reads a target type instance from a file.
         * @param path The path of the file to be read.
         * @return An instance of the target type.
         */
        virtual memory::pointer::unique_t<T> read(const std::filesystem::path&) const = 0;
    };

    /**
     * Creates a generic format reader for the given type.
     * @tparam T The type to get a generic reader from.
     * @return A format reader instance for the given type.
     */
    template <typename T>
    memory::pointer::unique_t<reader_t<T>> make_reader();

    /**
     * Reads a file and produces an instance of given type.
     * @tparam T The target type to read an instance of.
     * @param path The path of the file to read an instance from.
     * @return A pointer to an instance of target type.
     */
    template <typename T>
    MUSEQA_INLINE memory::pointer::unique_t<T> read(const std::filesystem::path& path)
    {
        const auto reader = make_reader<T>();
        return reader->read(path);
    }
}

MUSEQA_END_NAMESPACE
