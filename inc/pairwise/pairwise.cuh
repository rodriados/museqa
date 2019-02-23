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

namespace pairwise
{
    /**
     * The score of the alignment of a sequence pair.
     * @since 0.1.1
     */
    using Score = int32_t;

    /**
     * The aminoacid matches scoring tables are stored contiguously. Thus,
     * we explicitly declare their sizes.
     * @since 0.1.1
     */
    using ScoringTable = int8_t[25][25];

    /**
     * Stores the indeces of a pair of sequences to be aligned.
     * @since 0.1.1
     */
    struct Pair
    {
        uint16_t id[2];
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
    class Algorithm
    {
        protected:
            Buffer<Pair> pair;              /// The sequence pairs to be aligned.

        public:
            Algorithm() = default;
            Algorithm(const Algorithm&) = default;
            Algorithm(Algorithm&&) = default;

            virtual ~Algorithm() = default;

            Algorithm& operator=(const Algorithm&) = default;
            Algorithm& operator=(Algorithm&&) = default;

            virtual Buffer<Pair> generate(size_t);
            virtual Buffer<Score> run(const Configuration&) = 0;
    };

    /**
     * Functor responsible for representing an algorithm.
     * @see Pairwise::run
     * @since 0.1.1
     */
    using Factory = Functor<Algorithm *()>;

    /**
     * Manages all data and execution of the pairwise module.
     * @since 0.1.1
     */
    class Pairwise final
    {
        protected:
            Buffer<Score> score;        /// The buffer of all workpairs' scores.

        public:
            Pairwise() = default;
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

    namespace table
    {
        extern Pointer<ScoringTable> retrieve(const std::string&);
        extern Pointer<ScoringTable> toDevice(const std::string&);
    };
};

#endif