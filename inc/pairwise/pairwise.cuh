/**
 * Multiple Sequence Alignment pairwise header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PW_PAIRWISE_CUH_INCLUDED
#define PW_PAIRWISE_CUH_INCLUDED

#include <string>
#include <cstdint>

#include <cuda.cuh>
#include <utils.hpp>
#include <buffer.hpp>
#include <pointer.hpp>
#include <database.hpp>
#include <cartesian.hpp>
#include <exception.hpp>

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
     * Manages and encapsulates all configurable aspects of the pairwise module.
     * @since 0.1.1
     */
    struct configuration
    {
        const ::database& db;           /// The database of sequences to align.
        std::string algorithm;          /// The chosen pairwise algorithm.
        std::string table;              /// The chosen scoring table.
    };

    /**
     * Represents a pairwise module algorithm.
     * @since 0.1.1
     */    
    struct algorithm
    {
        buffer<pair> pairs;             /// The sequence pairs to be aligned.

        inline algorithm() noexcept = default;
        inline algorithm(const algorithm&) noexcept = default;
        inline algorithm(algorithm&&) noexcept = default;

        virtual ~algorithm() = default;

        inline algorithm& operator=(const algorithm&) = default;
        inline algorithm& operator=(algorithm&&) = default;

        virtual auto generate(size_t) -> buffer<pair>&;
        virtual auto run(const configuration&) -> buffer<score> = 0;
    };

    /**
     * Functor responsible for instantiating an algorithm.
     * @see pairwise::manager::run
     * @since 0.1.1
     */
    using factory = functor<algorithm *()>;

    /**
     * Manages all data and execution of the pairwise module.
     * @since 0.1.1
     */
    class manager final : public buffer<score>
    {
        protected:
            using element_type = score;                 /// The object's element type.
            using underlying_buffer = buffer<score>;    /// The object's underlying buffer.

        protected:
            size_t mcount;                              /// The total number of aligned sequences.

        public:
            inline manager() noexcept = default;
            inline manager(const manager&) noexcept = default;
            inline manager(manager&&) noexcept = default;

            inline manager& operator=(const manager&) = default;
            inline manager& operator=(manager&&) = default;

            using underlying_buffer::operator=;

            /**
             * Gives access to a pair score using a matrix format.
             * @param offset The requested element's offset.
             * @return The score of requested pair.
             */
            inline element_type operator[](const cartesian<2>& offset) const
            {
                const auto min = utils::min(offset[0], offset[1]);
                const auto max = utils::max(offset[0], offset[1]);
                return (min != max) ? underlying_buffer::operator[](utils::nchoose(max) + min) : 0;
            }

            /**
             * Informs the total number of sequences available in the module.
             * @return The number of processed sequences.
             */
            inline auto count() const noexcept -> size_t
            {
                return mcount;
            }

            static auto run(const configuration&) -> manager;

        protected:
            /**
             * Initializes a new manager instance with results obtained by the module.
             * @param buf The buffer with the module's results.
             * @param count The number of sequences aligned.
             */
            inline manager(const buffer<score>& buf, size_t count) noexcept
            :   underlying_buffer {buf}
            ,   mcount {count}
            {}
    };

    /**
     * The aminoacid substitution tables. These tables are stored contiguously
     * in memory, in order to facilitate accessing its elements.
     * @since 0.1.1
     */
    class scoring_table
    {
        public:
            using element_type = score;             /// The table content's type.
            using raw_type = element_type[25][25];  /// The raw table type.
            using pointer_type = pointer<raw_type>; /// The table's pointer type.

        protected:
            pointer_type mcontents;                 /// The table's contents.
            element_type mpenalty;                  /// The table's penalty value.

        public:
            inline scoring_table() noexcept = default;
            inline scoring_table(const scoring_table&) noexcept = default;
            inline scoring_table(scoring_table&&) noexcept = default;

            /**
             * Creates a new scoring table instance.
             * @param ptr The scoring table's pointer.
             * @param penalty The penalty applied by the scoring table.
             */
            inline scoring_table(const pointer_type& ptr, const element_type& penalty) noexcept
            :   mcontents {ptr}
            ,   mpenalty {penalty}
            {}

            inline scoring_table& operator=(const scoring_table&) = default;
            inline scoring_table& operator=(scoring_table&&) = default;

            /**
             * Gives access to the table's contents.
             * @param offset The requested table offset.
             * @return The required table row's reference.
             */
            __host__ __device__ inline element_type operator[](const cartesian<2>& offset) const noexcept
            {
                return (*mcontents)[offset[0]][offset[1]];
            }

            /**
             * Gives access to the table's penalty value.
             * @return The table's penalty value.
             */
            __host__ __device__ inline auto penalty() const noexcept -> decltype(mpenalty)
            {
                return mpenalty;
            }

            auto to_device() const -> scoring_table;

            static auto make(const std::string&) -> scoring_table;
            static auto list() noexcept -> const std::vector<std::string>&;
    };

    /**
     * Creates a module's configuration instance.
     * @param db The database of sequences to align.
     * @param algorithm The chosen pairwise algorithm.
     * @param table The chosen scoring table.
     * @return The module's configuration instance.
     */
    inline configuration configure(
            const ::database& db
        ,   const std::string& algorithm = {}
        ,   const std::string& table = {}
        )
    {
        return {db, algorithm, table};
    }
}

#endif