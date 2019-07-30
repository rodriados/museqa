/**
 * Multiple Sequence Alignment sequential needleman file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <cstdint>

#include "msa.hpp"
#include "node.hpp"
#include "buffer.hpp"
#include "pointer.hpp"
#include "encoder.hpp"
#include "database.hpp"

#include "pairwise/pairwise.cuh"
#include "pairwise/needleman.cuh"

using namespace pairwise;

namespace
{
    /**
     * Sequentially aligns two sequences using Needleman-Wunsch algorithm.
     * @param one The first sequence to align.
     * @param two The second sequence to align.
     * @param table The scoring table used to compare both sequences.
     * @return The alignment score.
     */
    static int32_t globalAlign(const Sequence& one, const Sequence& two, const ScoringTable& table)
    {
        const auto len1 = one.getLength();
        const auto len2 = two.getLength();
        const auto penalty = table.penalty;

        Buffer<int32_t> line = {new int32_t[len2 + 1], len2 + 1};
        int32_t done, val;

        // Filling 0-th line with penalties. This is the only initialization needed
        // for sequential algorithm.
        for(size_t i = 0; i <= len2; ++i)
            line[i] = - (penalty * i);

        for(size_t i = 0; i < len1; ++i) {
            // If the current line is at sequence end, then we can already finish the
            // algorithm, as no changes are expected to occur after the end of sequence.
            if(one[i] == encoder::end)
                break;

            // Initialize the 0-th column values. It will always be initialized with
            // penalties, in the same manner as the 0-th line.
            done = line[0];
            line[0] = - penalty * (i + 1);

            // Iterate over the second sequence, calculating the best alignment possible
            // for each of its characters.
            for(size_t j = 1, m = two.getLength(); j <= m; ++j) {
                val = two[j - 1] != encoder::end
                    ? utils::max(
                            done + table[one[i]][two[j - 1]]
                        ,   utils::max(line[j - 1] - penalty, line[j] - penalty)
                        )
                    : line[j - 1];

                done = line[j];
                line[j] = val;
            }
        }

        return line[two.getLength()];
    }

    /**
     * The sequential needleman algorithm object. This algorithm uses no
     * parallelism, besides pairs distribution to run the Needleman-Wunsch
     * algorithm.
     * @since 0.1.1
     */
    struct Sequential : public Needleman
    {
        /**
         * Executes the sequential Needleman-Wunsch algorithm for the pairwise step.
         * @param db The sequences available for alignment.
         * @param pairs The workpairs to align.
         * @param table The scoring table to use.
         * @return The score of aligned pairs.
         */
        Buffer<Score> alignDb(const ::Database& db, const ScoringTable& table)
        {
            const size_t total = this->pair.getSize();
            Buffer<Score> score {total};

            for(size_t i = 0; i < total; ) {
                const Sequence& seq1 = db[this->pair[i].id[0]];
                const Sequence& seq2 = db[this->pair[i].id[1]];

                score[i] = globalAlign(
                    seq1.getSize() > seq2.getSize() ? seq1 : seq2
                ,   seq1.getSize() > seq2.getSize() ? seq2 : seq1
                ,   table
                );
            }

            return score;
        }

        /**
         * Executes the sequential needleman algorithm for the pairwise step. This method is
         * responsible for distributing and gathering workload from different cluster nodes.
         * @param config The module's configuration.
         * @return The module's result value.
         */
        Buffer<Score> run(const Configuration& config) override
        {
            const ScoringTable table = ScoringTable::get(config.table);

            onlymaster this->generate(config.db.getCount());
            onlymaster msa::task("pairwise", "aligning %llu pairs", this->pair.getSize());

            this->scatter();
            onlyslaves this->score = alignDb(config.db, table);
            return this->gather();
        }
    };
};

/**
 * Instantiates a new sequential needleman instance.
 * @return The new algorithm instance.
 */
extern Algorithm *needleman::sequential()
{
    return new Sequential;
}
