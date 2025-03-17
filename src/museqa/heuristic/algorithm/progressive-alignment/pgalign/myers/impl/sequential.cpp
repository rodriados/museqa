/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Sequential implementation for the profile-aligner module's myers-miller algorithm.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#include "pgalign/pgalign.cuh"
#include "pgalign/myers/myers.cuh"

namespace
{
    using namespace museqa;
    using namespace pgalign;

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
            return alignment {};
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
