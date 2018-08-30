/**
 * Multiple Sequence Alignment needleman header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef PW_NEEDLEMAN_CUH_INCLUDED
#define PW_NEEDLEMAN_CUH_INCLUDED

#pragma once

#include <cstdint>
#include <vector>

#include "device.cuh"
#include "pairwise/sequence.cuh"
#include "pairwise/pairwise.hpp"

namespace pairwise
{
    /**
     * Composes with a pairwise::Pairwise instance to control the algorithm's execution.
     * @since 0.1.alpha
     */
    class Needleman
    {
        private:
            const Pairwise& pwise;          /// The pairwise instance for composition.
            std::vector<Workpair> pairs;    /// The sequence pairs to be processed.

        public:
            /**
             * Creates a new instance as a composition of Pairwise.
             * @param pwise The Pairwise instance to compose.
             */
            explicit Needleman(const Pairwise& pwise)
            :   pwise(pwise) {}

            void generate();
            void loadblosum();
            void run();

            void scatter();
            void gather() const;
    };

    /**
     * Holds all data to be sent to device for execution.
     * @since 0.1.alpha
     */
    class dNeedleman
    {
        protected:
            uint32_t count = 0;
            Workpair *pairs = nullptr;
            dSequenceList sequence;

        public:
            explicit dNeedleman(dSequenceList, const Workpair *, uint32_t);
            ~dNeedleman() noexcept;
    };

#ifdef __CUDACC__
    extern __global__ void needleman(dNeedleman, Score *);
#endif
};

#endif