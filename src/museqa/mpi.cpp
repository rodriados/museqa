/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file MPI wrapper global variables and functions definitions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#if !defined(MUSEQA_AVOID_MPI)

#include <map>
#include <cstdint>

#include <museqa/node.hpp>
#include <museqa/utility.hpp>
#include <museqa/exception.hpp>
#include <museqa/mpi/common.hpp>
#include <museqa/mpi/lambda.hpp>
#include <museqa/mpi/communicator.hpp>

namespace museqa
{
    using namespace mpi;

    /**
     * The default global communicator instance.
     * @since 1.0
     */
    decltype(mpi::world) mpi::world;

    /**
     * Informs the current node's global rank within the instantiated MPI topology.
     * @see museqa::mpi::init
     */
    const node::id& node::rank = world.rank;

    /**
     * Informs the total number of nodes within the current MPI topology.
     * @see museqa::mpi::init
     */
    const int32_t& node::count = world.size;

    /**
     * Stores the last collective operation execution's status.
     * @see museqa::mpi::probe
     * @since 1.0
     */
    mpi::status mpi::last_status;

    /**
     * Informs the currently active MPI operator function. This is necessary to
     * recover a wrapped function from within MPI execution.
     * @since 1.0
     */
    mpi::function::id mpi::function::active;

    /**
     * Maps a function identifier to its actual user-defined implementation. Unfortunately,
     * this is maybe the only reliable way to inject a wrapped function into the
     * operator actually called by MPI.
     * @since 1.0
     */
    std::map<mpi::function::id, void*> mpi::function::fmapper;

    /**
     * Initializes MPI's internal machinery and nodes communication.
     * @param argc The number of arguments received via command-line.
     * @param argv The program's command-line arguments.
     */
    void mpi::init(int& argc, char**& argv)
    {
        mpi::check(MPI_Init(&argc, &argv));

        mpi::communicator global {node::master, 1, memory::pointer::weak<void>{MPI_COMM_WORLD}};
        new (&world) mpi::communicator {global.duplicate()};
    }

    /**
     * Terminates MPI execution and cleans up all MPI state.
     * @see mpi::init
     */
    void mpi::finalize()
    {
        world.reset();
        mpi::check(MPI_Finalize());
    }

    /**
     * Builds a new communicator instance from a channel reference.
     * @param channel The channel reference to build a communicator from.
     * @return The new communicator instance.
     */
    mpi::communicator mpi::communicator::build(reference_type channel)
    {
        if (MPI_COMM_NULL != channel) {
            int32_t rank, size;
            mpi::check(MPI_Comm_rank(channel, &rank));
            mpi::check(MPI_Comm_size(channel, &size));
            mpi::check(MPI_Comm_set_errhandler(channel, MPI_ERRORS_RETURN));
            return mpi::communicator {rank, size, channel};
        } else {
            throw mpi::exception {mpi::error::comm, "invalid null communicator"};
        }
    }

    /**
     * Splits nodes within the communicator into different channels according to
     * each node's individual selection.
     * @param color The color selected by current node.
     * @param key The key used to assigned a node id in new communicator.
     * @return The communicator obtained from the split.
     */
    mpi::communicator mpi::communicator::split(int color, int key) const
    {
        reference_type channel;
        mpi::check(MPI_Comm_split(*this, color, (key > 0 ? key : rank), &channel));
        return build(channel);
    }

    /**
     * Duplicates the communicator with all its cached information.
     * @return The new duplicated communicator.
     */
    mpi::communicator mpi::communicator::duplicate() const
    {
        reference_type channel;
        mpi::check(MPI_Comm_dup(*this, &channel));
        return build(channel);
    }

    /**
     * Creates a new communicator channel from a previously create communicator
     * and a potentially smaller group of nodes.
     * @param comm The original communicator to create the new one from.
     * @param group The group of nodes to take part of the new communicator channel.
     * @param tag A tag for the new communicator channel.
     * @return The new communicator channel instance.
     */
    mpi::communicator mpi::communicator::create(
        const mpi::communicator& comm
      , const mpi::group& group
      , mpi::tag tag
    ) {
        reference_type channel;
        mpi::check(MPI_Comm_create_group(comm, group, utility::max(tag, 0), &channel));
        return build(channel);
    }

    /**
     * Creates a new node group from the union of two groups.
     * @param other A group instance to find the union with this instance.
     * @return The new node group instance.
     */
    mpi::group mpi::group::operator+(const mpi::group& other) const noexcept(!safe)
    {
        reference_type result;
        mpi::check(MPI_Group_union(*this, other, &result));
        return mpi::group {result};
    }

    /**
     * Creates a new node group from the difference between two groups.
     * @param other A group instance to find the difference with this instance.
     * @return The new node group instance.
     */
    mpi::group mpi::group::operator-(const mpi::group& other) const noexcept(!safe)
    {
        reference_type result;
        mpi::check(MPI_Group_difference(*this, other, &result));
        return mpi::group {result};
    }

    /**
     * Creates a new node group by excluding specific nodes from an existing group.
     * @param group The base group instance to have nodes excluded in new group.
     * @param nodes The list of excluded nodes in new group.
     * @param count The number of excluded nodes from the group.
     * @return The new node group instance.
     */
    mpi::group mpi::group::exclude(const mpi::group& group, const mpi::node *nodes, size_t count) noexcept(!safe)
    {
        reference_type result;
        mpi::check(MPI_Group_excl(group, static_cast<int>(count), nodes, &result));
        return mpi::group {result};
    }

    /**
     * Creates a new node group by selecting specific nodes to form a new group.
     * @param group The base group instance to pick nodes for new group from.
     * @param nodes The list of nodes to be included in new group.
     * @param count The total number of nodes in the new group.
     * @return The new node group instance.
     */
    mpi::group mpi::group::include(const mpi::group& group, const mpi::node *nodes, size_t count) noexcept(!safe)
    {
        reference_type result;
        mpi::check(MPI_Group_incl(group, static_cast<int>(count), nodes, &result));
        return mpi::group {result};
    }

    /**
     * Creates a new node group from the intersection between two existing nodes.
     * @param fst The first group to the intersection operation.
     * @param snd The second group to the intersection operation.
     * @return The new node group instance.
     */
    mpi::group mpi::group::intersection(const mpi::group& fst, const mpi::group& snd) noexcept(!safe)
    {
        reference_type result;
        mpi::check(MPI_Group_intersection(fst, snd, &result));
        return mpi::group {result};
    }
}

#endif
