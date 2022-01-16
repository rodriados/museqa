/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Configuration and inclusion of the argagg third party library.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>

#if !defined(MUSEQA_AVOID_ARGAGG)
  #include <argagg/argagg.hpp>
#endif
