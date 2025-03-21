/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Configuration and inclusion of the reflector third party library.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>

#ifndef MUSEQA_AVOID_THIRDPARTY_REFLECTOR
  #ifdef MUSEQA_OVERRIDE_REFLECTOR
    #include MUSEQA_OVERRIDE_REFLECTOR
  #elif __has_include(<rodriados/reflector.h>)
    #include <rodriados/reflector.h>
  #elif __has_include(<reflector/api.h>)
    #include <reflector/api.h>
  #else
    #include <reflector.h>
  #endif
#endif
