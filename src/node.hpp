/** 
 * Multiple Sequence Alignment node header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _NODE_HPP_
#define _NODE_HPP_

/**
 *  This struct holds relevant data about the current MPI processes.
 *  @since 0.1.alpha
 */
struct NodeInfo final
{
    int rank;         /// The current MPI process rank.
    int size;         /// The total number of MPI processes.
};

/*
 * Declaring global variables.
 */
extern NodeInfo nodeinfo;

/*
 * Defines some process control macros. These macros are to be used when
 * it is needed to check whether the current process is master or not.
 */
#define __master        (0)
#define __ismaster()    (nodeinfo.rank == __master)
#define __isslave()     (nodeinfo.rank != __master)
#define __onlymaster    if(__ismaster())
#define __onlyslaves    if(__isslave())
#define __onlyslave(i)  if(__isslave() && nodeinfo.rank == (i))

#endif