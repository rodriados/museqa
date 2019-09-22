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

#include "cuda.cuh"
#include "utils.hpp"
#include "buffer.hpp"
#include "pointer.hpp"
#include "database.hpp"
#include "exception.hpp"

namespace pairwise
{
    /**
     * The score of a sequence pair alignment.
     * @since 0.1.1
     */
    using Score = int32_t;

    /**
     * Represents a reference for a sequence.
     * @since 0.1.1
     */
    using SequenceRef = uint16_t;

    /**
     * Stores the indeces of a pair of sequences to be aligned.
     * @since 0.1.1
     */
    struct Pair
    {
        SequenceRef id[2];
    };

    /**
     * Manages and encapsulates all configurable aspects of the pairwise module.
     * @since 0.1.1
     */
    struct Configuration
    {
        const ::Database& db;           /// The database of sequences to align.
        std::string algorithm;          /// The chosen pairwise algorithm.
        std::string table;              /// The chosen scoring table.
    };

    /**
     * Represents a pairwise module algorithm.
     * @since 0.1.1
     */    
    struct Algorithm
    {
        Buffer<Pair> pair;              /// The sequence pairs to be aligned.

        inline Algorithm() noexcept = default;
        inline Algorithm(const Algorithm&) noexcept = default;
        inline Algorithm(Algorithm&&) noexcept = default;

        virtual ~Algorithm() = default;

        inline Algorithm& operator=(const Algorithm&) = default;
        inline Algorithm& operator=(Algorithm&&) = default;

        virtual Buffer<Pair> generate(size_t);
        virtual Buffer<Score> run(const Configuration&) = 0;
    };

    /**
     * Functor responsible for instantiating an algorithm.
     * @see Pairwise::run
     * @since 0.1.1
     */
    using Factory = Functor<Algorithm *()>;

    /**
     * Manages all data and execution of the pairwise module.
     * @since 0.1.1
     */
    class Pairwise final : public Buffer<Score>
    {
        protected:
            size_t count;           /// The total number of sequences available.

        public:
            inline Pairwise() noexcept = default;
            inline Pairwise(const Pairwise&) noexcept = default;
            inline Pairwise(Pairwise&&) noexcept = default;

            inline Pairwise& operator=(const Pairwise&) = default;
            inline Pairwise& operator=(Pairwise&&) = default;

            using Buffer<Score>::operator=;

            /**
             * Gives access to a pair score using a matrix format.
             * @param x The requested x-axis offset.
             * @param y The requested y-axis offset.
             * @return The score of requested pair.
             */
            inline Score operator()(ptrdiff_t x, ptrdiff_t y) const
            {
                const auto min = utils::min(x, y);
                const auto max = utils::max(x, y);
                return (x == y) ? 0 : Buffer<Score>::operator[](utils::combinations(max) + min);
            }

            /**
             * Informs the total number of sequences available in the module.
             * @return The number of processed sequences.
             */
            inline size_t getCount() const noexcept
            {
                return count;
            }

            void run(const Configuration&);
    };

    /**
     * The aminoacid substitution tables. These tables are stored contiguously
     * in memory, in order to facilitate accessing its elements.
     * @since 0.1.1
     */
    struct ScoringTable
    {
        using Element = int8_t;
        using RawTable = Element[25][25];

        Pointer<RawTable> contents;     /// The table's contents.
        Element penalty;                /// The table's penalty value.

        inline ScoringTable() noexcept = default;
        inline ScoringTable(const ScoringTable&) noexcept = default;
        inline ScoringTable(ScoringTable&&) noexcept = default;

        /**
         * Creates a new scoring table instance.
         * @param ptr The scoring table's pointer.
         * @param penalty The penalty applied by the scoring table.
         */
        inline ScoringTable(const Pointer<RawTable>& ptr, const Element& penalty) noexcept
        :   contents {ptr}
        ,   penalty {penalty}
        {}

        inline ScoringTable& operator=(const ScoringTable&) = default;
        inline ScoringTable& operator=(ScoringTable&&) = default;

        /**
         * Gives access to the table's contents.
         * @param offset The requested table offset.
         * @return The required table row's reference.
         */
        __host__ __device__ inline auto operator[](ptrdiff_t offset) const noexcept
        -> const Element (&)[25]
        {
            return (*contents)[offset];
        }

        static ScoringTable get(const std::string&);
        static ScoringTable toDevice(const std::string&);
        static const std::vector<std::string>& getList() noexcept;
    };

    /**
     * Creates a module's configuration instance.
     * @param db The database of sequences to align.
     * @param algorithm The chosen pairwise algorithm.
     * @param table The chosen scoring table.
     * @return The module's configuration instance.
     */
    inline Configuration configure
        (   const ::Database& db
        ,   const std::string& algorithm = {}
        ,   const std::string& table     = {}   )
    {
        return {db, algorithm, table};
    }
};

#endif