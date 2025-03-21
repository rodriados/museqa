# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file Script responsible for finding the cuda-wrappers project.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2025-present Rodrigo Siqueira
cmake_minimum_required(VERSION 3.24)

include(FetchContent)

set(CUDAWRAPPERS_REPOSITORY "https://github.com/eyalroz/cuda-api-wrappers.git")
set(CUDAWRAPPERS_REPOSITORY_TAG "v0.8.2")

# Declares the remote source of the required package and allows it to be found.
# If needed, the package will be downloaded and cached for build.
FetchContent_Declare(
  cuda-api-wrappers
    GIT_SHALLOW true
    GIT_REPOSITORY ${CUDAWRAPPERS_REPOSITORY}
    GIT_TAG ${CUDAWRAPPERS_REPOSITORY_TAG})

# Now that the package is declared, we must find and configure it so that its variables
# and targets are made available for the parent context.
FetchContent_MakeAvailable(cuda-api-wrappers)
