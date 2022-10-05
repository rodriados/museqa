/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Language dialect-specific configurations and macro definitions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

/*
 * Discovers the C++ language dialect in use for the current compilation. A specific
 * dialect might not be supported or might be required for certain functionalities
 * to work properly.
 */
#if defined(__cplusplus)
  #if __cplusplus < 201103L
    #define MUSEQA_CPP_DIALECT 2003
  #elif __cplusplus < 201402L
    #define MUSEQA_CPP_DIALECT 2011
  #elif __cplusplus < 201703L
    #define MUSEQA_CPP_DIALECT 2014
  #elif __cplusplus == 201703L
    #define MUSEQA_CPP_DIALECT 2017
  #elif __cplusplus > 201703L
    #define MUSEQA_CPP_DIALECT 2020
  #endif
#endif
