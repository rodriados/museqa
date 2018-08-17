/**
 * Multiple Sequence Alignment pairwise header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _PW_PAIRWISE_CUH_
#define _PW_PAIRWISE_CUH_

#include <cstdint>

#include "fasta.hpp"
#include "pairwise/sequence.cuh"

namespace pairwise
{
    /**
     * Indicates a pair of sequences to be aligned.
     * @since 0.1.alpha
     */
    struct Workpair
    {
        uint16_t first;     /// Index of the first sequence index to align
        uint16_t second;    /// Index of the second sequence index to align
    };

    /**
     * Stores score information about a sequence pair.
     * @since 0.1.alpha
     */
    struct Score
    {
        int32_t score = 0;          /// The cached score value for a sequence pair.
        uint16_t matches = 0;       /// The number of matches in the pair.
        uint16_t mismatches = 0;    /// The number of mismatches in the pair.
        uint16_t gaps = 0;          /// The number of gaps in the pair.
    };

    /**
     * Manages the pairwise module execution.
     * @since 0.1.alpha
     */
    class Pairwise
    {
        private:
            Score *score;
            SequenceList list;

        public:
            Pairwise(const Fasta&);
            ~Pairwise() noexcept;

            /**
             * Informs the number of sequences processed.
             * @return The number of seqeunces processed.
             */
            inline uint16_t getCount() const
            {
                return this->list.getCount();
            }

            /**
             * Gives access to a processed pair score instance.
             * @return The requested pair score instance.
             */
            inline const Score& getScore(uint16_t x, uint16_t y) const
            {
                return this->score[x * this->getCount() + y];
            }

            void process();
    };
};

#endif