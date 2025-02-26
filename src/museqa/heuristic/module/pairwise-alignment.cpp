/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implementation of pairwise-alignment heuristic module.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#include <map>
#include <string>
#include <stdexcept>

#include <museqa/environment.h>
#include <museqa/pipeline.hpp>
#include <museqa/memory/pointer.hpp>
#include <museqa/utility/functor.hpp>

#include <museqa/heuristic/module/pairwise.cuh>
#include <museqa/heuristic/algorithm/pairwise-alignment/exception.hpp>
#include <museqa/heuristic/algorithm/pairwise-alignment/matrix.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace pairwise = heuristic::algorithm::pairwise;

namespace heuristic::module
{
    /**
     * Factory type for instantiating a pairwise-module algorithm implementation.
     * The instance must inherit from pairwise-module's abstract algorithm type.
     * @since 1.0
     */
    using factory_t = utility::functor_t<memory::pointer::shared_t<pairwise_t::algorithm_t>()>;

    /**
     * The mapping between the module's algorithm implementations and their respective
     * names. A single implementation may be referenced more than once.
     * @since 1.0
     */
    static const std::map<std::string, factory_t> factory_dispatcher = {};

    /**
     * Executes the module's main task and algorithm on the pipeline.
     * @param pipe The pipeline's transitive state instance.
     */
    void pairwise_t::run(pipeline::pipe_t& pipe) const
    try {
        const auto context = algorithm_t::context_t {
            m_params
        };

        const factory_t& factory = factory_dispatcher.at(m_params.input.algorithm);
        const memory::pointer::shared_t<algorithm_t> worker = factory ();

        auto result = factory::memory::pointer::shared<matrix_t>();
            *result = worker->run(context);

        pipe->set(pairwise_t::matrix, result);
    } catch (const std::out_of_range& e) {
        throw pairwise::exception_t("unknown pairwise algorithm: " + m_params.input.algorithm);
    }
}

MUSEQA_END_NAMESPACE
