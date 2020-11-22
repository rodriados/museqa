/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implementation for the heuristic's bootstrap module.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#include <vector>

#include "mpi.hpp"
#include "utils.hpp"
#include "tuple.hpp"
#include "encoder.hpp"
#include "pointer.hpp"
#include "database.hpp"
#include "sequence.hpp"

#include "bootstrap.hpp"

namespace museqa
{
    namespace
    {
        /**
         * Loads all sequence database files from command line arguments.
         * @param io The IO service instance to get file names from.
         * @return The sequences database with all loaded sequences.
         */
        static auto load(const io::service& io) -> pointer<database>
        {
            auto db = pointer<database>::make();
            auto dblist = io.load<database>();

            for(const auto& current : dblist)
                db->merge(current);

            return db;
        }

        /**
         * Unpacks a database from its flattened components and rebuilds an instance
         * with the exact sequence contents as before serialization.
         * @param sizes The list of serialized sequences sizes.
         * @param blocks The flattened database blocks.
         * @return The reconstructed database.
         */
        static auto unpack(
                const std::vector<size_t>& sizes
            ,   const std::vector<encoder::block>& blocks
            )
        -> pointer<database>
        {
            auto db = pointer<database>::make();

            for(size_t i = 0, j = 0, n = sizes.size(); i < n; ++i) {
                db->add(sequence::copy(&blocks[j], sizes[i]));
                j += sizes[i];
            }

            return db;
        }
    }

    namespace module
    {
        /**
         * Runs the bootstrap module. This method shall solely load the sequence
         * database from files distribute it to all cluster nodes.
         * @param io The pipeline's IO service instance.
         * @return A conduit with the module's processed results.
         */
        auto bootstrap::run(const io::service& io, const bootstrap::pipe&) const -> bootstrap::pipe
        {
            pointer<database> db;
            std::vector<size_t> sizes;
            std::vector<encoder::block> blocks;

            onlymaster db = load(io);

            onlymaster for(const auto& entry : *db) {
                size.push_back(entry.contents.size());
                block.insert(block.end(), entry.contents.begin(), entry.contents.end());
            }

            sizes = mpi::broadcast(sizes);
            blocks = mpi::broadcast(blocks);

            onlyslaves db = unpack(sizes, blocks);

            auto ptr = new bootstrap::conduit {db};
            mpi::barrier();

            return bootstrap::pipe {ptr};
        }
    }
}
