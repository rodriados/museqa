/**
 * Multiple Sequence Alignment sequential neighbor-joining file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019-2020 Rodrigo Siqueira
 */
#include <cstdint>

#include <phylogeny/phylogeny.cuh>
#include <phylogeny/njoining.cuh>

namespace
{
    using namespace msa;
    using namespace phylogeny;

    /**
     * The sequential needleman algorithm object. This algorithm uses no
     * parallelism whatsoever.
     * @since 0.1.1
     */
    struct sequential : public njoining::algorithm
    {
        /**
         * Executes the sequential neighbor-joining algorithm for the phylogeny
         * step. This method is responsible for coordinating the execution.
         * @return The module's result value.
         */
        auto run(const context& ctx) -> tree override
        {
            this->rootless = std::vector<otu>();
        }
    };
}

namespace msa
{
    /**
     * Instantiates a new sequential neighbor-joining instance.
     * @return The new algorithm instance.
     */
    extern auto phylogeny::njoining::sequential() -> phylogeny::algorithm *
    {
        return new ::sequential;
    }
}
