/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Configuration and inclusion of the CUDA wrapper third party library.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>

#if !defined(MUSEQA_AVOID_CUDA)
  #if !defined(MUSEQA_AVOID_CUDAWRAPPERS)
    #include <cuda/api.hpp>
  #else
    #include <cuda.h>
    #include <cuda_runtime_api.h>
  #endif
#endif

#if !defined(MUSEQA_AVOID_CUDAWRAPPERS)

namespace cuda::device::current
{
    /**
     * Waits for all previously-scheduled tasks on all streams on the currently
     * active device to conclude.
     * @see cuda::synchronize
     */
    inline void synchronize()
    {
        auto device = cuda::device::current::get();
        cuda::synchronize(device);
    }
}

#endif
