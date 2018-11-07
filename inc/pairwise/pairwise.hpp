/**
 * Multiple Sequence Alignment pairwise header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef PW_PAIRWISE_HPP_INCLUDED
#define PW_PAIRWISE_HPP_INCLUDED

#pragma once

#include <cstdint>
#include <vector>

#include "msa.hpp"
#include "fasta.hpp"
#include "buffer.hpp"
#include "pointer.hpp"
#include "pairwise/sequence.cuh"

/*
 * Defining some configuration macros. These can be changed if needed.
 */
#define pw_prefer_shared_mem 1
#define pw_batches_per_block 15
#define pw_threads_per_block 32

namespace pairwise
{
    /**
     * Indicates a pair of sequences to be aligned.
     * @since 0.1.alpha
     */
    struct Workpair
    {
        uint16_t first;
        uint16_t second;
    };

    /**
     * The score of a sequence pair alignment.
     * @since 0.1.alpha
     */
    using Score = int32_t;

    /**
     * Abstract class to represent an algorithm. The concrete algorithm class will
     * be composed to the Pairwise class.
     * @since 0.1.alpha
     */
    class Algorithm
    {
        protected:
            SequenceList list;                      /// The list of sequences to align.
            Buffer<Score>& score;                   /// The buffer of score values;
            std::vector<Workpair> pair;             /// The vector of pairs to compare.
            SharedPointer<int8_t[25][25]> table;    /// The scoring table to be used.
            int8_t penalty;                         /// The gap penalty.

        public:
            Algorithm() = default;
            Algorithm(const Algorithm&) = default;
            Algorithm(Algorithm&&) = default;

            /**
             * Creates a new algorithm instance.
             * @param pwise The pairwise instance.
             */
            inline Algorithm(const SequenceList& list, Buffer<Score>& score)
            :   list(list)
            ,   score(score)
            {
                onlymaster this->generate();
                this->loadBlosum();
            }

            virtual ~Algorithm() noexcept = default;

            Algorithm& operator=(const Algorithm&) = default;
            Algorithm& operator=(Algorithm&&) = default;

            virtual void generate();
            virtual void run() = 0;

            virtual void scatter() = 0;
            virtual void gather() = 0;

            void loadBlosum();
    };

    /**
     * Manages the pairwise module execution.
     * @since 0.1.alpha
     */
    class Pairwise final
    {
        protected:
            SequenceList list;      /// The list of sequences to align.
            Buffer<Score> score;    /// The buffer of score values;

        public:
            Pairwise() = default;
            Pairwise(const Pairwise&) = default;
            Pairwise(Pairwise&&) = default;

            Pairwise(const Fasta&);

            ~Pairwise() noexcept = default;

            Pairwise& operator=(const Pairwise&) = default;
            Pairwise& operator=(Pairwise&&) = default;

            /**
             * Informs the number of pairs processed or to process.
             * @return The number of pairs this instance shall process.
             */
            inline size_t getCount() const
            {
                return this->score.getSize();
            }

            /**
             * Gives access to the list of sequences to process.
             * @return The sequence list to process.
             */
            inline const SequenceList& getList() const
            {
                return this->list;
            }

            /**
             * Accesses score value of workpair.
             * @return The requested pair score value.
             */
            inline const Score getScore(const Workpair& pair) const
            {
                return this->getScore(pair.first, pair.second);
            }

            /**
             * Accesses a score according to its offset.
             * @return The requested pair score value.
             */
            inline const Score getScore(ptrdiff_t offset) const
            {
                return this->score[offset];
            }

            /**
             * Gives access to a processed pair score value.
             * @return The requested pair score value.
             */
            inline const Score getScore(uint16_t x, uint16_t y) const
            {
                if(x == y)  return 0;

                uint16_t min, max;

                if(x > y) { max = x; min = y; }
                else      { max = y; min = x; }

                return this->score[((max + 1) * max) / 2 + min];
            }
    };
};

#endif