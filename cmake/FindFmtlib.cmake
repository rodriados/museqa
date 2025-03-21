# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file Script responsible for finding the fmtlib project.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2025-present Rodrigo Siqueira
cmake_minimum_required(VERSION 3.24)

include(FetchContent)

set(FMTLIB_REPOSITORY "https://github.com/fmtlib/fmt.git")
set(FMTLIB_REPOSITORY_TAG "12.1.0")

set(FMT_INSTALL ON)

# Declares the remote source of the required package and allows it to be found.
# If needed, the package will be downloaded and cached for build.
FetchContent_Declare(
  fmt
    GIT_SHALLOW true
    GIT_REPOSITORY ${FMTLIB_REPOSITORY}
    GIT_TAG ${FMTLIB_REPOSITORY_TAG})

# Now that the package is declared, we must find and configure it so that its variables
# and targets are made available for the parent context.
FetchContent_MakeAvailable(fmt)
