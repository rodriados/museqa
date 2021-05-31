/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Cluster node identifiers and value definitions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>

namespace museqa
{
    namespace node
    {
        /**
         * The type for identifying an specific cluster node within the cluster
         * itself or within a node group.
         * @since 1.0
         */
        using id = int32_t;

        /**#@+
         * Keeps track of the total number of nodes that are currently allocated
         * to the current execution, and the global identification of the current
         * node within the cluster.
         * @see museqa::mpi::init
         */
        #if !defined(MUSEQA_AVOID_MPI)
            extern node::id rank;
            extern int32_t count;
        #else
            enum : node::id { rank = 0 };
            enum : int32_t { count = 1 };
        #endif
        /**#@-*/

        /*
         * Defining the master node rank value. It is recommended not to change
         * the master node's rank, as no other node rank is garanteed to exist.
         * @since 1.0
         */
        enum : node::id { master = 0 };
    }
}

/*
 * Defines some node control macros. These macros are to be used when it is needed
 * to check whether the current node is the master or a work node.
 * @since 1.0
 */
#if !defined(MUSEQA_AVOID_MPI)
  #define onlymaster   if(museqa::node::rank == museqa::node::master)
  #define onlyslaves   if(museqa::node::rank != museqa::node::master)
  #define onlynode(i)  if((i) == museqa::node::rank)
#else
  #define onlymaster   if(1)
  #define onlyslaves   if(1)
  #define onlynode(i)  if((i) == museqa::node::master)
#endif
