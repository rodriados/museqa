/** 
 * Multiple Sequence Alignment node header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _NODE_HPP_
#define _NODE_HPP_

/*
 * Defining macro indicating the node rank to be used as the master node. It
 * is recommended not to change it, as no other rank is garanteed to exist.
 */
#define __msa_master_node_id__ 0

namespace node
{
    /*
     * Declaring global variables
     */
    extern int rank;
    extern int size;

    /**
     * Informs whether the current node is the master node.
     * @return Is this node the master?
     */
    inline bool ismaster()
    {
        return rank == __msa_master_node_id__;
    }

    /**
     * Informs whether the current node is a slave node.
     * @return Is this node a slave?
     */
    inline bool isslave()
    {
        return rank != __msa_master_node_id__;
    }
};

/*
 * Defines some process control macros. These macros are to be used when
 * it is needed to check whether the current process is master or not.
 */
#define __onlymaster    if(node::ismaster())
#define __onlyslaves    if(node::isslave())
#define __onlyslave(i)  if(node::isslave() && node::rank == (i))

#endif