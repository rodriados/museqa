/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Cluster node identifiers and value definitions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>

#include <museqa/environment.h>

MUSEQA_BEGIN_NAMESPACE

namespace node
{
    /**
     * The type for identifying a specific cluster node within the cluster itself
     * or within a subgroup of nodes.
     * @since 1.0
     */
    using id = int32_t;

    /**#@+
     * Keeps track of the total number of nodes that are currently allocated to
     * the current execution, and the global identification of the current node
     * within the cluster.
     * @see museqa::mpi::init
     * @since 1.0
     */
  #if !defined(MUSEQA_AVOID_MPI)
    extern const node::id& rank;
    extern const int32_t& count;
  #else
    enum : node::id { rank = 0 };
    enum : int32_t { count = 1 };
  #endif
    /**#@-*/

    /*
     * Defining the master node rank value. It is recommended not to change the
     * master node's rank, as no other node rank is garanteed to exist.
     * @since 1.0
     */
    enum : node::id { master = 0 };
}

MUSEQA_END_NAMESPACE

/*
 * Defines node control macros. These macros are to be used when it is needed to
 * check whether the current node is the master, a compute or has a specific rank.
 * @since 1.0
 */
#if !defined(MUSEQA_AVOID_MPI)
  #define __museqanode__(i)     (museqa::node::rank == (i))
  #define __museqamaster__      (__museqanode__(museqa::node::master))
  #define __museqacompute__     (!__museqamaster__)
#else
  #define __museqanode__(i)     (museqa::node::master == (i))
  #define __museqamaster__      (true)
  #define __museqacompute__     (true)
#endif

/*
 * Defines macros for program flow control which condition depends on the rank of
 * node currently in execution.
 * @since 1.0
 */
#define __museqaonlynode__(i)   if (__museqanode__(i))
#define __museqaonlymaster__    if (__museqamaster__)
#define __museqaonlycompute__   if (__museqacompute__)

#if !defined(__onlynode__) && !defined(__onlymaster__) && !defined(__onlycompute__)
  #define __onlynode__(i)   __museqaonlynode__(i)
  #define __onlymaster__    __museqaonlymaster__
  #define __onlycompute__   __museqaonlycompute__
#endif
