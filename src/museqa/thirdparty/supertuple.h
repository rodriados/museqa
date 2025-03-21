/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Configuration and inclusion of the supertuple third party library.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>

#ifndef MUSEQA_AVOID_THIRDPARTY_SUPERTUPLE
  #ifdef MUSEQA_OVERRIDE_SUPERTUPLE
    #include MUSEQA_OVERRIDE_SUPERTUPLE
  #elif __has_include(<rodriados/supertuple.h>)
    #include <rodriados/supertuple.h>
  #elif __has_include(<supertuple/api.h>)
    #include <supertuple/api.h>
  #else
    #include <supertuple.h>
  #endif
#endif
