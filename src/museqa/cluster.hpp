/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Cluster runtime identifiers and constants definitions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>
#include <museqa/environment.h>

MUSEQA_BEGIN_NAMESPACE

namespace cluster
{
    /**
     * The type of a process identifier within the cluster of MPI processes.
     * @since 1.0
     */
    using pid = int32_t;

    /**#@+
     * Keeps track of the current process identifier and the total number of processes
     * within the cluster in which the runtime is executing.
     * @see mpiwcpp17::initialize
     * @since 1.0
     */
  #if !defined(MUSEQA_AVOID_MPI)
    extern const cluster::pid& rank;
    extern const int32_t& size;
  #else
    enum : cluster::pid { rank = 0 };
    enum : int32_t { size = 1 };
  #endif
    /**#@-*/

    /**
     * Definition of the master cluster process PID value. It is recommended not
     * to change the master process's PID, as no other PID is guaranteed to exist.
     * @since 1.0
     */
    enum : cluster::pid { master = 0 };

    /**
     * Evaluates whether the current process is the one with the given PID.
     * @param pid The PID to check whether corresponds to the current process.
     * @return Does the current process match the requested PID?
     */
    inline constexpr bool onlyprocess(cluster::pid pid) noexcept
    {
      #if !defined(MUSEQA_AVOID_MPI)
        return pid == cluster::rank;
      #else
        return true;
      #endif
    }

    /**
     * Evaluates whether the current process is the master process within cluster.
     * @return Is the current process the cluster's master process?
     */
    inline constexpr bool onlymaster() noexcept
    {
      #if !defined(MUSEQA_AVOID_MPI)
        return cluster::onlyprocess(cluster::master);
      #else
        return true;
      #endif
    }

    /**
     * Evaluates whether the current process is a compute process within cluster.
     * @return Is the current process one of the cluster's compute processes?
     */
    inline constexpr bool onlycompute() noexcept
    {
      #if !defined(MUSEQA_AVOID_MPI)
        return !cluster::onlymaster();
      #else
        return true;
      #endif
    }
}

MUSEQA_END_NAMESPACE
