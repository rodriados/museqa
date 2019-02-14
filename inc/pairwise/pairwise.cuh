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

#include "utils.hpp"
#include "buffer.hpp"
#include "pointer.hpp"
#include "database.hpp"

#include "pairwise/database.cuh"

/*
 * Configuration macros. These values interfere directly into the algorithm's
 * exection, and should be modified with certain caution.
 */
#define pw_threads_per_block 32 
#define pw_prefer_shared_memory 1

namespace pairwise
{
    /**
     * Stores the indeces of a pair of sequences to be aligned.
     * @since 0.1.1
     */
    struct Workpair
    {
        uint16_t id[2];
    };

    /**
     * The score of the alignment of a sequence pair.
     * @since 0.1.1
     */
    using Score = int32_t;

    /**
     * Manages and encapsulates all configurable aspects of the pairwise module.
     * @since 0.1.1
     */
    struct Configuration
    {
        const ::Database& db;           /// The database of sequences to align.
        std::string algorithm;          /// The chosen pairwise algorithm.
        std::string scoringTable;       /// The chosen scoring table.
    };

    /**
     * An abstract pairwise algorithm class.
     * @since 0.1.1
     */
    class Algorithm
    {
        private:
            const Configuration& config;    /// The module configuration.

        protected:
            Buffer<Score> score;            /// The buffer of calculated workpair scores.
            Buffer<Workpair> pair;          /// The list of pairs to align.
            pairwise::Database db;          /// The compressed database of sequences.

        public:
            Algorithm() = default;
            Algorithm(const Algorithm&) = default;
            Algorithm(Algorithm&&) = default;

            /**
             * Instantiates a new algorithm instance.
             * @param config The module configuration.
             */
            inline Algorithm(const Configuration& config)
            :   config {config}
            {}

            virtual ~Algorithm() noexcept = default;

            Algorithm& operator=(const Algorithm&) = default;
            Algorithm& operator=(Algorithm&&) = default;

            virtual Buffer<Workpair> generate() = 0;
            virtual Buffer<Score> run() = 0;
    };

    /**
     * Functor responsible for building a new algorithm instance.
     * @see Pairwise::run
     * @since 0.1.1
     */
    using AlgorithmFactory = Functor<Algorithm *(const Configuration&)>;

    /**
     * The aminoacid matches scoring tables are stored contiguously. Thus,
     * we explicitly declare their sizes.
     * @since 0.1.1
     */
    using ScoringTable = int8_t[25][25];

    /**
     * Manages all data and execution of the pairwise module.
     * @since 0.1.1
     */
    class Pairwise final
    {
        protected:
            Buffer<Score> score;        /// The buffer of all workpairs' scores.

        public:
            Pairwise();
            Pairwise(const Pairwise&) = default;
            Pairwise(Pairwise&&) = default;

            Pairwise& operator=(const Pairwise&) = default;
            Pairwise& operator=(Pairwise&&) = default;

            /**
             * Gives access to a specific workpair score.
             * @param offset The requested workpair score.
             * @return The retrieved score.
             */
            inline const Score& operator[](ptrdiff_t offset) const
            {
                return score[offset];
            }

            /**
             * Gives access to all workpairs' scores.
             * @return The score buffer's internal pointer.
             */
            inline Score *getBuffer() const
            {
                return score.getBuffer();
            }

            /**
             * Informs the number of processed pairs or to process.
             * @return The number of pairs this instance shall process.
             */
            inline size_t getCount() const
            {
                return score.getSize();
            }

            void run(const Configuration&);
    };

    namespace scoring
    {
        extern AutoPointer<ScoringTable> get(const std::string&);
        extern AutoPointer<ScoringTable> toDevice(const std::string&);
    };
};

#endif