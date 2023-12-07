/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implementation of pairwise algorithm's resulting distance matrix.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <vector>

#include <museqa/environment.h>
#include <museqa/utility.hpp>
#include <museqa/guard.hpp>

#include <museqa/memory/buffer.hpp>
#include <museqa/heuristic/algorithm/pairwise.cuh>
#include <museqa/heuristic/algorithm/pairwise/exception.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace heuristic::algorithm::pairwise
{
    /**
     * The distance matrix of pairwise alignments between biological sequences.
     * At last, the matrix is the resulting object produced by the module.
     * @since 1.0
     */
    class matrix : protected memory::buffer_t<pairwise::score_t>
    {
        protected:
            typedef memory::buffer_t<pairwise::score_t> underlying_t;

        protected:
            size_t m_count = 0;

        public:
            inline matrix() noexcept = default;
            inline matrix(const matrix&) noexcept = default;
            inline matrix(matrix&&) noexcept = default;

            /**
             * Initializes a new distance matrix from a buffer linearly containing
             * pairwise distances between a set of sequences.
             * @param buffer The linear distances' buffer to copy into the matrix.
             */
            inline matrix(const underlying_t& buffer) __museqasafe__
              : underlying_t (buffer)
              , m_count (utility::oeis::a002024(m_capacity))
            {
                museqa::guard<pairwise::exception_t>(
                    m_capacity == (size_t) utility::oeis::a000217(m_count)
                  , "number of pair distances is not compatible with number of sequences"
                );
            }

            inline matrix& operator=(const matrix&) noexcept = default;
            inline matrix& operator=(matrix&&) noexcept = default;

            /**
             * Retrieves the pairwise distance of a sequence pair on the matrix.
             * @param pair The identifier of a sequence pair.
             * @return The pairwise distance between a pair of sequences.
             */
            inline auto operator[](const pairwise::pair_t& pair) const -> pairwise::score_t
            {
                const auto max = utility::max(pair.a, pair.b);
                const auto min = utility::min(pair.a, pair.b);

                return (pair.a != pair.b)
                    ? underlying_t::operator[](min + utility::nchoose(max))
                    : pairwise::score_t(0);
            }

            /**
             * Informs the total number of sequences represented on the matrix.
             * @return The number of sequences with pairwise alignments.
             */
            inline auto count() const noexcept -> size_t
            {
                return m_count;
            }
    };
}

MUSEQA_END_NAMESPACE
