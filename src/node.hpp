/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements MPI cluster's node identifiers and values.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>

#include "utils.hpp"
#include "environment.h"

namespace museqa
{
    namespace node
    {
        /**
         * Identifies a node in a cluster or in a group.
         * @since 0.1.1
         */
        using id = int32_t;

        /**#@+
         * Informs the number of total nodes in the cluster and the current node's
         * global identification.
         * @see mpi::init
         */
        #if !defined(__museqa_runtime_cython)
            extern node::id rank;
            extern int32_t count;
        #else
            constexpr node::id rank = 0;
            constexpr int32_t count = 1;
        #endif
        /**#@-*/

        /*
         * Defining the master node rank value. It is recommended not to change
         * the master node's rank, as no other node rank is garanteed to exist.
         */
        enum : node::id { master = 0 };
    }
}

/*
 * Defines some node control macros. These macros are to be used when it is needed
 * to check whether the current node is the master or not.
 */
#if !defined(__museqa_runtime_cython)
  #define onlymaster   if(museqa::node::rank == museqa::node::master)
  #define onlyslaves   if(museqa::node::rank != museqa::node::master)
  #define onlynode(i)  if((i) == museqa::node::rank)
#else
  #define onlymaster   if(1)
  #define onlyslaves   if(1)
  #define onlynode(i)  if((i) == museqa::node::master)
#endif
