/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file CUDA wrapper global variables and functions definitions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#if !defined(MUSEQA_AVOID_CUDA)

#include <cuda.h>

#include <string>
#include <cstdint>

#include <museqa/cuda/common.cuh>

namespace museqa
{
    using namespace cuda;

    /**
     * Retrieves a brief textual explanation for a specified kind of CUDA
     * runtime API status or error code.
     * @param code The error code to be described.
     * @return The error description.
     */
    inline auto cuda::error::describe(error::code code) noexcept -> std::string
    {
        return cudaGetErrorString(code);
    }
}

#endif
