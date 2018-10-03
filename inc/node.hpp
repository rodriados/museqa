/** 
 * Multiple Sequence Alignment node header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef NODE_HPP_INCLUDED
#define NODE_HPP_INCLUDED

#pragma once

/*
 * Defining macro indicating the node rank to be used as the master node. It
 * is recommended not to change it, as no other rank is garanteed to exist.
 */
#define master_node_id 0

namespace cluster
{
    /*
     * Declaring global variables
     */
    extern int size;
    extern int rank;
    static constexpr const int master = master_node_id;
};

namespace node
{
    /**
     * Informs whether the current node has given rank.
     * @param rank The rank to be checked.
     * @return Does this node has given rank?
     */        
    inline bool is(int rank)
    {
        return cluster::rank == rank;
    }

    /**
     * Informs whether the current node is the master node.
     * @return Is this node the master?
     */
    inline bool ismaster()
    {
        return is(cluster::master);
    }

    /**
     * Informs whether the current node is a slave node.
     * @return Is this node a slave?
     */
    inline bool isslave()
    {
        return !is(cluster::master);
    }
};

/*
 * Defines some process control macros. These macros are to be used when
 * it is needed to check whether the current process is master or not.
 */
#define onlymaster    if(node::ismaster())
#define onlyslaves    if(node::isslave())
#define onlyslave(i)  if(node::isslave() && node::is(i))

#undef master_node_id

#endif