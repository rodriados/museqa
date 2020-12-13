/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements the pairwise module's functionality.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <vector>
#include <cstdint>

#include "cuda.cuh"
#include "utils.hpp"
#include "point.hpp"
#include "buffer.hpp"
#include "matrix.hpp"
#include "functor.hpp"
#include "pointer.hpp"
#include "database.hpp"

namespace museqa
{
    namespace pairwise
    {
        /**
         * The score of a sequence pair alignment.
         * @since 0.1.1
         */
        using score = float;

        /**
         * Represents a reference for a sequence.
         * @since 0.1.1
         */
        using seqref = int_least32_t;

        /**
         * Stores the indeces of a pair of sequences to be aligned.
         * @since 0.1.1
         */
        union pair
        {
            struct { seqref first, second; };
            seqref id[2];
        };

        /**
         * Represents a pairwise distance matrix. At last, this object represents
         * the pairwise module's execution's final result.
         * @since 0.1.1
         */
        class distance_matrix : protected buffer<score>
        {
            protected:
                using underlying_type = buffer<score>;
                using point_type = typename matrix<score>::point_type;

            public:
                using element_type = score;         /// The distance matrix's element type.

            protected:
                size_t m_count;                     /// The number of sequences represented in the matrix.

            public:
                inline distance_matrix() noexcept = default;
                inline distance_matrix(const distance_matrix&) noexcept = default;
                inline distance_matrix(distance_matrix&&) noexcept = default;

                /**
                 * Instantiates a new distance matrix from a buffer linearly containing
                 * the pairwise distances between all sequences.
                 * @param buf The linear buffer of pairwise distances.
                 * @param count The total number of sequences represented.
                 */
                inline distance_matrix(const underlying_type& buf, size_t count) noexcept
                :   underlying_type {buf}
                ,   m_count {count}
                {}

                inline distance_matrix& operator=(const distance_matrix&) = default;
                inline distance_matrix& operator=(distance_matrix&&) = default;

                /**
                 * Retrieves the pairwise distance of a specific pair on the matrix.
                 * @param offset The requested pair's offset.
                 * @return The pairwise distance between sequences in given pair.
                 */
                inline auto operator[](const point_type& offset) const -> element_type
                {
                    const auto max = utils::max(offset[0], offset[1]);
                    const auto min = utils::min(offset[0], offset[1]);

                    return (max > min)
                        ? underlying_type::operator[](utils::nchoose(max) + min)
                        : element_type {0};
                }

                /**
                 * Informs the total number of pairwise aligned sequences.
                 * @return The number of sequences processed by the module.
                 */
                inline auto count() const noexcept -> size_t
                {
                    return m_count;
                }
        };

        /**
         * An aminoacid or nucleotide substitution table. This table is used to
         * calculate the match value between two aminoacid or nucleotide characters.
         * @since 0.1.1
         */
        class scoring_table
        {
            public:
                using element_type = score;                 /// The table content's type.
                using raw_type = element_type[25][25];      /// The raw table type.
                using pointer_type = pointer<raw_type>;     /// The table's pointer type.

            protected:
                pointer_type m_contents;                    /// The table's contents.
                element_type m_penalty;                     /// The table's penalty value.

            public:
                inline scoring_table() noexcept = default;
                inline scoring_table(const scoring_table&) noexcept = default;
                inline scoring_table(scoring_table&&) noexcept = default;

                /**
                 * Creates a new scoring table instance.
                 * @param ptr The scoring table's pointer.
                 * @param penalty The penalty applied by the scoring table.
                 */
                __host__ __device__ inline scoring_table(
                        const pointer_type& ptr
                    ,   const element_type& penalty
                    ) noexcept
                :   m_contents {ptr}
                ,   m_penalty {penalty}
                {}

                __device__ scoring_table(const pointer_type&, const scoring_table&) noexcept;

                inline scoring_table& operator=(const scoring_table&) = default;
                inline scoring_table& operator=(scoring_table&&) = default;

                /**
                 * Gives access to the table's contents.
                 * @param offset The requested table offset.
                 * @return The required table row's reference.
                 */
                __host__ __device__ inline element_type operator[](const point<2>& offset) const noexcept
                {
                    return (*m_contents)[offset.x][offset.y];
                }

                /**
                 * Gives access to the table's penalty value.
                 * @return The table's penalty value.
                 */
                __host__ __device__ inline auto penalty() const noexcept -> element_type
                {
                    return m_penalty;
                }

                scoring_table to_device() const;

                static auto has(const std::string&) -> bool;
                static auto make(const std::string&) -> scoring_table;
                static auto list() noexcept -> const std::vector<std::string>&;
        };

        /**
         * Represents a common pairwise algorithm context.
         * @since 0.1.1
         */
        struct context
        {
            const museqa::database& db;
            const scoring_table& table;
        };

        /**
         * Functor responsible for instantiating an algorithm.
         * @see pairwise::run
         * @since 0.1.1
         */
        using factory = functor<struct algorithm *()>;

        /**
         * Represents a pairwise module algorithm.
         * @since 0.1.1
         */
        struct algorithm
        {
            inline algorithm() noexcept = default;
            inline algorithm(const algorithm&) noexcept = default;
            inline algorithm(algorithm&&) noexcept = default;

            virtual ~algorithm() = default;

            inline algorithm& operator=(const algorithm&) = default;
            inline algorithm& operator=(algorithm&&) = default;

            virtual auto generate(size_t) const -> buffer<pair>;
            virtual auto run(const context&) const -> distance_matrix = 0;

            static auto has(const std::string&) -> bool;
            static auto make(const std::string&) -> const factory&;
            static auto list() noexcept -> const std::vector<std::string>&;
        };

        /**
         * Runs the module when not on a pipeline.
         * @param db The database of sequences to align.
         * @param table The chosen scoring table.
         * @param algorithm The chosen pairwise algorithm.
         * @return The chosen algorithm's resulting distance matrix.
         */
        inline distance_matrix run(
                const museqa::database& db
            ,   const scoring_table& table
            ,   const std::string& algorithm = "default"
            )
        {
            auto lambda = pairwise::algorithm::make(algorithm);
            
            const pairwise::algorithm *worker = lambda ();
            auto result = worker->run({db, table});
            
            delete worker;
            return result;
        }
    }
}
