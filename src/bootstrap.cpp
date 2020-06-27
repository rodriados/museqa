/**
 * Multiple Sequence Alignment bootstrap module file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020 Rodrigo Siqueira
 */
#include <vector>

#include <mpi.hpp>
#include <utils.hpp>
#include <tuple.hpp>
#include <encoder.hpp>
#include <pointer.hpp>
#include <database.hpp>
#include <sequence.hpp>
#include <bootstrap.hpp>

namespace msa
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
         * Serializes a database instance to a tuple with the database's flattened
         * contents and its repective sequences' lengths. The sequences' descriptions,
         * though, are not serialized and thus their contents are lost.
         * @param db The database to be serializable.
         * @return The serialized database's contents.
         */
        static auto serialize(const pointer<database>& db) -> decltype(auto)
        {
            std::vector<size_t> size;
            std::vector<encoder::block> block;

            for(const auto& entry : *db) {
                size.push_back(entry.contents.size());
                block.insert(block.end(), entry.contents.begin(), entry.contents.end());
            }

            return tuple<
                    std::vector<size_t>
                ,   std::vector<encoder::block>
                > {size, block};
        }

        /**
         * Unserializes a database from its flattened components and rebuilds an
         * instance with the exact sequence contents as before serialization.
         * @param sizes The list of serialized sequences sizes.
         * @param blocks The flattened database blocks.
         * @return The reconstructed database.
         */
        static auto unserialize(
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

    namespace bootstrap
    {
        /**
         * Runs the bootstrap module. This method shall solely load the sequence
         * database from files distribute it to all cluster nodes.
         * @param io The pipeline's IO service instance.
         * @return A conduit with the module's processed results.
         */
        auto module::run(const io::service& io, const module::pipe&) const -> module::pipe
        {
            pointer<database> db;
            std::vector<size_t> sizes;
            std::vector<encoder::block> blocks;

            onlymaster {
                db = load(io);
                utils::tie(sizes, blocks) = serialize(db);
            }

            sizes = mpi::broadcast(sizes);
            blocks = mpi::broadcast(blocks);

            onlyslaves {
                db = unserialize(sizes, blocks);
            }

            auto ptr = new module::conduit {db};
            mpi::barrier();

            return module::pipe {ptr};
        }
    }
}
