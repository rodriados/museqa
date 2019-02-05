/** 
 * Multiple Sequence Alignment node header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef NODE_HPP_INCLUDED
#define NODE_HPP_INCLUDED

#include <cstdint>

namespace node
{
#ifndef msa_compile_cython
    extern uint16_t size;
    extern uint16_t rank;
#endif

    /*
     * Defining macro indicating the node rank to be used as the master node. It
     * is recommended not to change it, as no other rank is garanteed to exist.
     */
    enum : uint16_t { master = 0 };
};

/*
 * Defines some process control macros. These macros are to be used when
 * it is needed to check whether the current process is master or not.
 */
#ifndef msa_compile_cython
  #define onlymaster   if(node::rank == node::master)
  #define onlyslaves   if(node::rank != node::master)
  #define onlynode(i)  if((i) == node::rank)
#else
  #define onlymaster   if(1)
  #define onlyslaves   if(0)
  #define onlynode(i)  if((i) == node::master)
#endif

#endif