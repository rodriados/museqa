/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file CUDA device utilities and helper functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#if !defined(MUSEQA_AVOID_CUDA)

#include <museqa/cuda/common.cuh>

namespace museqa
{
    namespace cuda
    {
        namespace device
        {
            /**
             * The CUDA device identification type. Here, we simply define a device
             * as its numeric identifier, which is usefull for breaking dependencies
             * between distinct objects and easier interation with code using the
             * original CUDA APIs.
             * @since 1.0
             */
            using id = cuda::word;

            /**
             * If the CUDA runtime has not yet been set to a specific device, this
             * is the device id it defaults to.
             * @since 1.0
             */
            enum : id { default_device = 0 };
        }
    }
}

#endif
