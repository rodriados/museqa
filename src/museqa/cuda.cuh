/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Imports the whole codebase of CUDA function and object wrappers.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#if !defined(MUSEQA_AVOID_CUDA)

#include <museqa/cuda/common.cuh>
#include <museqa/cuda/device.cuh>
#include <museqa/cuda/stream.cuh>
#include <museqa/cuda/event.cuh>
#include <museqa/cuda/memory.cuh>

#endif
