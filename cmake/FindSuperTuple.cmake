# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file Script responsible for finding the SuperTuple project.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2025-present Rodrigo Siqueira
cmake_minimum_required(VERSION 3.24)

include(FetchContent)

set(SUPERTUPLE_REPOSITORY "https://github.com/rodriados/supertuple.git")
set(SUPERTUPLE_REPOSITORY_TAG "v1.1.4")

# Declares the remote source of the required package and allows it to be found.
# If needed, the package will be downloaded and cached for build.
FetchContent_Declare(
  SuperTuple
    GIT_SHALLOW true
    GIT_REPOSITORY ${SUPERTUPLE_REPOSITORY}
    GIT_TAG ${SUPERTUPLE_REPOSITORY_TAG})

# Now that the package is declared, we must find and configure it so that its variables
# and targets are made available for the parent context.
FetchContent_MakeAvailable(SuperTuple)
