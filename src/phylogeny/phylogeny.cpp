/**
 * Multiple Sequence Alignment phylogeny module file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019 Rodrigo Siqueira
 */
#include <map>
#include <string>

#include <msa.hpp>
#include <exception.hpp>

#include <phylogeny/phylogeny.cuh>
#include <phylogeny/njoining.cuh>

/*
 * Keeps the list of available algorithms and their respective factories.
 * @since 0.1.1
 */
static const std::map<std::string, phylogeny::factory> dispatcher = {
    /*{"njoining",                phylogeny::njoining::hybrid}
,   {"njoining-hybrid",         phylogeny::njoining::hybrid}
,*/ {"njoining-sequential",     phylogeny::njoining::sequential}
/*,   {"njoining-distributed",    phylogeny::njoining::distributed}*/
};

/**
 * Executes a phylogeny algorithm, transforming the distance matrix between the
 * sequences into a pseudo-phylogenetic tree.
 * @param config The module's configuration.
 * @return The new module manager instance.
 */
auto phylogeny::manager::run(const phylogeny::configuration& config) -> phylogeny::manager
{
    const auto& selected = dispatcher.find(config.algorithm);

    enforce(selected != dispatcher.end(), "unknown phylogeny algorithm <bold>%s</>", config.algorithm);
    onlymaster watchdog::info("chosen phylogeny algorithm <bold>%s</>", config.algorithm);

    onlymaster watchdog::init("phylogeny", "building the phylogenetic tree");
    phylogeny::algorithm *worker = (selected->second)();
    phylogeny::manager result {worker->run(config)};
    onlymaster watchdog::finish("phylogeny", "phylogenetic tree fully constructed");
    delete worker;

    return result;
}
