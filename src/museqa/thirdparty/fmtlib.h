/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Configuration and inclusion of the fmtlib third party library.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>

#ifndef MUSEQA_AVOID_THIRDPARTY_FMTLIB
  MUSEQA_DISABLE_NVCC_WARNING_BEGIN(128)
  MUSEQA_DISABLE_NVCC_WARNING_BEGIN(2417)
    #include <fmt/format.h>
    #include <fmt/ranges.h>
  MUSEQA_DISABLE_NVCC_WARNING_END(2417)
  MUSEQA_DISABLE_NVCC_WARNING_END(128)
#endif
