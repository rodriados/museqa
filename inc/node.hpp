/** 
 * Multiple Sequence Alignment node header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef NODE_HPP_INCLUDED
#define NODE_HPP_INCLUDED

#include <utils.hpp>

namespace node
{
    /**
     * Identifies a node in a cluster or in a group.
     * @since 0.1.1
     */
    using id = int32_t;

    #if !defined(__cython__)
        /**#@+
         * Informs the number of total nodes in the cluster and the current node's
         * global identification.
         * @see mpi::init
         */
        extern id& rank;
        extern uint32_t& count;
        /**#@-*/
    #endif

    /*
     * Defining the master node rank value. It is recommended not to change the
     * master node's rank, as no other node rank is garanteed to exist.
     */
    enum : id { master = 0 };
}

/*
 * Defines some process control macros. These macros are to be used when
 * it is needed to check whether the current process is master or not.
 */
#if !defined(__cython__)
  #define onlymaster   if(node::rank == node::master)
  #define onlyslaves   if(node::rank != node::master)
  #define onlynode(i)  if((i) == node::rank)
#else
  #define onlymaster   if(1)
  #define onlyslaves   if(1)
  #define onlynode(i)  if((i) == node::master)
#endif

#endif