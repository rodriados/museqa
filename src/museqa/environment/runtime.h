/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Runtime definition and configuration.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

/*
 * Defines whether the compilation context is related to the host or device runtime.
 * Code availability or functionality might vary depending on the runtime it is
 * expected to run on.
 */
#if defined(__NVCOMPILER_CUDA__)
  #define MUSEQA_RUNTIME_DEVICE __builtin_is_device_code()
  #define MUSEQA_RUNTIME_HOST (!__builtin_is_device_code())
  #define MUSEQA_INCLUDE_DEVICE_CODE 1
  #define MUSEQA_INCLUDE_HOST_CODE 1
#elif defined(__CUDA_ARCH__)
  #define MUSEQA_RUNTIME_DEVICE 1
  #define MUSEQA_RUNTIME_HOST 0
  #define MUSEQA_INCLUDE_DEVICE_CODE 1
  #define MUSEQA_INCLUDE_HOST_CODE 0
#else
  #define MUSEQA_RUNTIME_DEVICE 0
  #define MUSEQA_RUNTIME_HOST 1
  #define MUSEQA_INCLUDE_DEVICE_CODE 0
  #define MUSEQA_INCLUDE_HOST_CODE 1
#endif
