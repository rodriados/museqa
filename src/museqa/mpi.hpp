/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Imports the whole codebase around MPI, such wrappers and helper functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#if !defined(MUSEQA_AVOID_MPI)
  #include <museqa/mpi/common.hpp>
  #include <museqa/mpi/type.hpp>
  #include <museqa/mpi/status.hpp>
  #include <museqa/mpi/lambda.hpp>
  #include <museqa/mpi/communicator.hpp>
  #include <museqa/mpi/collective.hpp>
#endif
