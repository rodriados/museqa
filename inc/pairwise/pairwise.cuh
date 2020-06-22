/**
 * Multiple Sequence Alignment pairwise header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2020 Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <vector>
#include <cstdint>

#include <io.hpp>
#include <cuda.cuh>
#include <utils.hpp>
#include <buffer.hpp>
#include <pointer.hpp>
#include <database.hpp>
#include <pipeline.hpp>
#include <bootstrap.hpp>
#include <cartesian.hpp>
#include <symmatrix.hpp>

namespace msa
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
        using seqref = int_least16_t;

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
        class distance_matrix : public symmatrix<score>
        {
            public:
                using element_type = score;                 /// The distance matrix's element type.

            protected:
                using underlying_matrix = symmatrix<score>; /// The distance matrix's underlying type.

            public:
                inline distance_matrix() noexcept = default;
                inline distance_matrix(const distance_matrix&) noexcept = default;
                inline distance_matrix(distance_matrix&&) noexcept = default;

                inline distance_matrix& operator=(const distance_matrix&) = default;
                inline distance_matrix& operator=(distance_matrix&&) = default;

                using underlying_matrix::operator=;

                /**
                 * Creates a new distance matrix by inflating a linear buffer with
                 * the corresponding matrix's distances.
                 * @param buf The buffer containing the matrix's distances.
                 * @param count The total number of sequences aligned.
                 * @return A new inflated distance matrix instance.
                 */
                static inline auto inflate(const buffer<score>& buf, size_t count) noexcept
                -> distance_matrix
                {
                    return distance_matrix {buf, count};
                }

            private:
                /**
                 * Constructs a new distance matrix by inflating a distances buffer.
                 * @param buf The buffer containing the matrix's distances.
                 * @param count The total number of sequences aligned.
                 */
                inline explicit distance_matrix(const buffer<score>& buf, size_t count) noexcept
                :   underlying_matrix {underlying_matrix::make(count)}
                {
                    for(size_t i = 0, n = 0; i < count; ++i)
                        for(size_t j = i; j < count; ++j)
                            operator[]({i, j}) = (i != j) ? buf[n++] : 0;
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
                __host__ __device__ inline element_type operator[](const cartesian<2>& offset) const noexcept
                {
                    return (*m_contents)[offset[0]][offset[1]];
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
            const msa::database& db;
            const scoring_table& table;
        };

        /**
         * Functor responsible for instantiating an algorithm.
         * @see pairwise::manager::run
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
         * Defines the module's conduit. This conduit is composed of the sequences
         * to be aligned and their pairwise distance matrix.
         * @since 0.1.1
         */
        struct conduit : public pipeline::conduit
        {
            const msa::database db;                 /// The loaded sequences' database.
            const distance_matrix distances;        /// The sequences' pairwise distances.
            const size_t total;                     /// The total number of sequences.

            inline conduit() noexcept = delete;
            inline conduit(const conduit&) = default;
            inline conduit(conduit&&) = default;

            /**
             * Instantiates a new conduit.
             * @param db The sequence database to transfer to the next module.
             * @param dmat The database's resulting pairwise distance matrix.
             */
            inline conduit(const msa::database& db, const distance_matrix& dmat) noexcept
            :   db {db}
            ,   distances {dmat}
            ,   total {db.count()}
            {}

            inline conduit& operator=(const conduit&) = delete;
            inline conduit& operator=(conduit&&) = delete;
        };

        /**
         * Defines the module's pipeline manager. This object will be the one responsible
         * for checking and managing the module's execution when on a pipeline.
         * @since 0.1.1
         */
        struct module : public pipeline::module
        {
            using previous = bootstrap::module;     /// Indicates the expected previous module.
            using conduit = pairwise::conduit;      /// The module's conduit type.

            using pipe = pointer<pipeline::conduit>;
            
            /**
             * Returns an string identifying the module's name.
             * @return The module's name.
             */
            inline auto name() const -> const char * override
            {
                return "pairwise";
            }

            auto run(const io::service&, const pipe&) const -> pipe override;
            auto check(const io::service&) const -> bool override;
        };

        /**
         * Runs the module when not on a pipeline.
         * @param db The database of sequences to align.
         * @param table The chosen scoring table.
         * @param algorithm The chosen pairwise algorithm.
         * @return The chosen algorithm's resulting distance matrix.
         */
        inline distance_matrix run(
                const msa::database& db
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
