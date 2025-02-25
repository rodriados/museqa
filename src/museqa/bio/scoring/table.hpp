/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Scoring tables and substitution matrices types and functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>
#include <museqa/utility.hpp>
#include <museqa/guard.hpp>

#include <museqa/bio/alphabet.hpp>
#include <museqa/bio/scoring/score.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace bio::scoring
{
    /**
     * An aminoacid or nucleotide substitution table. It contains the scores for
     * matches and mismatches symbols of biological sequence alignments. The table
     * is triangular, so that the ordering of sequences should not change the score.
     * @since 1.0
     */
    class table_t
    {
        public:
            MUSEQA_CONSTEXPR static auto symbol_count = alphabet::count;

        private:
            score_t m_matrix[utility::a000217(symbol_count)] = {};

        public:
            /**
             * Gets the score for matching two sequences symbols.
             * @param a The first symbol to be matched.
             * @param b The second symbol to be matched.
             * @return The resulting match score.
             */
            MUSEQA_CUDA_INLINE score_t& m(
                alphabet::symbol_t a
              , alphabet::symbol_t b
            ) MUSEQA_SAFE_EXCEPT {
                guard(a < symbol_count && b < symbol_count
                  , "symbol has no representation in scoring matrix");
                const auto x = utility::min(a, b);
                const auto y = utility::max(a, b);
                return m_matrix[x + utility::a000217(y)];
            }
    };
}

MUSEQA_END_NAMESPACE
