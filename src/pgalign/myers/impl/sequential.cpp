/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Sequential implementation for the profile-aligner module's myers-miller algorithm.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#include <stack>

#include "node.hpp"
#include "buffer.hpp"
#include "database.hpp"
#include "phylogeny.cuh"

#include "pgalign/pgalign.cuh"
#include "pgalign/sequence.cuh"
#include "pgalign/alignment.cuh"
#include "pgalign/myers/myers.cuh"

namespace
{
    using namespace museqa;
    using namespace pgalign;

    using node_type = guidetree::node_type;

    static alignment align_profiles(const profile& one, const profile& two, const scoring_table& table)
    {
        
    }

    static const node_type *node_child(const guidetree& tree, const node_type *node, int child_id)
    {
        const auto child = node->child[child_id];
        return child != guidetree::undefined ? &tree[child] : nullptr;
    }

    static alignment initialize(const museqa::database& db, const guidetree& tree, size_t count)
    {
        size_t leaf = 0;

        std::stack<const node_type *> stack;
        const node_type *current = &tree.root();
        auto order = buffer<pgalign::sequence>::make(count);

        while(current != nullptr || !stack.empty()) {
            while(current != nullptr) {
                stack.push(current);
                current = node_child(tree, current, 0);
            }

            current = stack.top();
            
            if(current->level == 0)
                order[leaf++] = pgalign::sequence {db[current->id].contents};

            current = node_child(tree, current, 1);
            stack.pop();
        }

        return alignment {order};
    }

    /**
     * The sequential myers-miller algorithm object. This algorithm uses no GPU
     * devices parallelism whatsoever.
     * @since 0.1.1
     */
    struct sequential : public myers::algorithm
    {
        /**
         * Executes the sequential myers-miller algorithm for the profile-aligner
         * step. This method is responsible for distributing and gathering workload
         * from different cluster nodes.
         * @param context The algorithm's context.
         * @return The module's result value.
         */
        auto run(const context& ctx) const -> alignment override
        {
            auto result = initialize(ctx.db, ctx.tree, ctx.count);
            return result;
        }
    };
}

namespace museqa
{
    /**
     * Instantiates a new sequential myers-miller algorithm instance.
     * @return The new algorithm instance.
     */
    extern auto pgalign::myers::sequential() -> pgalign::algorithm *
    {
        return new ::sequential;
    }
}
