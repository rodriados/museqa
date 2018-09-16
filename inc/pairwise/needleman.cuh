/**
 * Multiple Sequence Alignment needleman header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef PW_NEEDLEMAN_CUH_INCLUDED
#define PW_NEEDLEMAN_CUH_INCLUDED

#pragma once

#include "buffer.hpp"
#include "pointer.hpp"
#include "pairwise/sequence.cuh"
#include "pairwise/pairwise.hpp"

namespace pairwise
{
    /**
     * Implements the needleman algorithm for pairwise step.
     * @since 0.1.alpha
     */
    class Needleman : public Algorithm
    {
        public:
            /**
             * Creates a new algorithm instance.
             * @param list The sequence list to process.
             * @param score The score array reference to store results.
             */
            explicit Needleman(const SequenceList& list, Buffer<Score>& score)
            :   Algorithm(list, score) {}

            void run() override;

        protected:
            void scatter();
            void gather();
    };

#ifdef __CUDACC__
    namespace needleman
    {
        /**
         * Holds all data to be sent to device for execution.
         * @since 0.1.alpha
         */
        struct Input
        {
            //Buffer<Workpair> pair;                  // The list of workpairs to process.
            //dSequenceList sequence;                 // The list of sequences to process.
            SharedPointer<int8_t[25][25]> table;    // The scoring table to be used.
        };

        extern __global__ void exec(Input /*, Buffer<Score>*/);
    };
#endif
};

#endif