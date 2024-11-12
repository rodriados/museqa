/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Scoring tables and substitution matrices types and functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>
#include <museqa/geometry.hpp>
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
             * @param ref The pair of symbols to be matched.
             * @return The resulting match score.
             */
            MUSEQA_CUDA_INLINE score_t& operator[](
                geometry::point_t<2, alphabet::symbol_t> ref
            ) MUSEQA_SAFE_EXCEPT {
                guard(ref.a < symbol_count && ref.b < symbol_count
                  , "symbol has no representation in scoring matrix");
                const auto a = utility::min(ref.a, ref.b);
                const auto b = utility::max(ref.a, ref.b);
                return m_matrix[a + utility::a000217(b)];
            }
    };
}

MUSEQA_END_NAMESPACE
