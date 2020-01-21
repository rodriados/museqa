/** 
 * Multiple Sequence Alignment MPI helper file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#include <map>
#include <vector>

#include <mpi.hpp>
#include <node.hpp>

namespace msa
{
    /**
     * The default communicator instance.
     * @see mpi::Communicator
     */
    mpi::communicator mpi::world;

    /**
     * Keeps track of all generated datatypes throughout execution.
     * @since 0.1.1
     */
    std::vector<mpi::datatype::id> mpi::datatype::ref_type;

    /**
     * Keeps track of all user defined operators created during execution.
     * @since 0.1.1
     */
    std::vector<mpi::op::id> mpi::op::ref_op;

    /**
     * Maps a datatype to an user-created operator. This is necessary because
     * it is almost technically impossible to inject the operator inside the
     * wrapper without an extremelly convoluted mechanism.
     * @since 0.1.1
     */
    std::map<mpi::op::id, void *> mpi::op::op_list;

    /**
     * Informs the currently active operator. This will be useful for injecting
     * the correct operator inside the wrapper.
     * @since 0.1.1
     */
    mpi::op::id mpi::op::active;

    /**#@+
     * Node identification values in cluster.
     * @see mpi::init
     */
    const node::id& node::rank = mpi::world.rank;
    const int32_t& node::count = mpi::world.size;
    /**#@-*/

    using payload = mpi::payload::base;

    /**
     * Initializes a new communicator.
     * @param ref The internal MPI communicator reference.
     */
    mpi::communicator::communicator(const mpi::communicator::raw_type& ref)
    :   m_ref {ref}
    {
        mpi::check(MPI_Comm_rank(ref, &rank));
        mpi::check(MPI_Comm_size(ref, &size));
    }

    /**
     * Synchronizes all nodes in a communicator.
     */
    void mpi::communicator::barrier() const
    {
        mpi::check(MPI_Barrier(m_ref));
    }

    /**
     * Broadcasts data to all nodes in given communicator.
     * @param load The payload to be broadcast to all nodes.
     * @param root The operation's root node.
     */
    void mpi::communicator::broadcast(payload *load, const mpi::node& root) const
    {
        mpi::check(MPI_Bcast(load->raw(), load->size(), load->type(), root, m_ref));
    }

    /**
     * Sends data to a node connected to the cluster.
     * @param load The payload to be sent to destiny node.
     * @param dest The payload's destination node.
     * @param tag The identifying message tag.
     */
    void mpi::communicator::send(payload *load, const mpi::node& dest, const mpi::tag& tag) const
    {
        mpi::check(MPI_Send(load->raw(), load->size(), load->type(), dest, tag, m_ref));
    }

    /**
     * Receives data from a node connected to the cluster.
     * @param load The payload to receive the incoming message.
     * @param src The payload's source node.
     * @param tag The identifying message tag.
     */
    auto mpi::communicator::receive(payload *load, const mpi::node& src, const mpi::tag& tag) const -> mpi::status
    {
        mpi::status::raw_type status;
        mpi::check(MPI_Recv(load->raw(), load->size(), load->type(), src, tag, m_ref, &status));
        return {status};
    }

    /**
     * Inspects incoming message and retrieves its status.
     * @param src The source node.
     * @param tag The identifying message tag.
     * @return The inspected message status.
     */
    auto mpi::communicator::probe(const mpi::node& src, const mpi::tag& tag) const -> mpi::status
    {
        mpi::status::raw_type status;
        mpi::check(MPI_Probe(src, tag, m_ref, &status));
        return {status};
    }

    /**
     * Gather data from nodes according to given distribution.
     * @param send The outgoing payload buffer to be sent to root.
     * @param recv The incoming payload buffer to receive from all nodes.
     * @param root The operation's root node.
     */
    void mpi::communicator::gather(payload *send, payload *recv, const mpi::node& root) const
    {
        mpi::check(MPI_Gather(
                send->raw(), send->size(), send->type()
            ,   recv->raw(), recv->size(), recv->type()
            ,   root
            ,   m_ref
            ));
    }

    /**
     * Gather data from nodes according to given distribution and displacement list.
     * @param send The outgoing payload buffer to be sent to root.
     * @param recv The incoming payload buffer to receive from all nodes.
     * @param displ The displacement of each incoming message from nodes.
     * @param root The operation's root node.
     */
    void mpi::communicator::gatherv(payload *send, payload *recv, int *displ, const mpi::node& root) const
    {
        mpi::check(MPI_Gatherv(
                send->raw(), send->size(), send->type()
            ,   recv->raw(), recv->size(), displ, recv->type()
            ,   root
            ,   m_ref
            ));
    }

    /**
     * Initializes the cluster's communication and identifies the node in the cluster.
     * @param argc The number of arguments sent from terminal.
     * @param argv The arguments sent from terminal.
     */
    void mpi::init(int& argc, char **& argv)
    {
        mpi::check(MPI_Init(&argc, &argv));
        mpi::world = mpi::communicator::build(MPI_COMM_WORLD);
    }

    /**
     * Finalizes all cluster communication operations between nodes.
     * @see mpi::init
     */
    void mpi::finalize()
    {
        for(mpi::datatype::id& typeref : mpi::datatype::ref_type)
            mpi::check(MPI_Type_free(&typeref));
     
        for(mpi::op::id& opref : mpi::op::ref_op)
            mpi::check(MPI_Op_free(&opref));
     
        MPI_Finalize();
    }
}