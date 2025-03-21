# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file Script responsible for finding the MPIwCpp17 project.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2025-present Rodrigo Siqueira
cmake_minimum_required(VERSION 3.24)

include(FetchContent)

set(MPIWCPP17_REPOSITORY "https://github.com/rodriados/mpiwcpp17.git")
set(MPIWCPP17_REPOSITORY_TAG "master")

# Declares the remote source of the required package and allows it to be found.
# If needed, the package will be downloaded and cached for build.
FetchContent_Declare(
  MPIwCpp17
    GIT_SHALLOW true
    GIT_REPOSITORY ${MPIWCPP17_REPOSITORY}
    GIT_TAG ${MPIWCPP17_REPOSITORY_TAG})

# Now that the package is declared, we must find and configure it so that its variables
# and targets are made available for the parent context.
FetchContent_MakeAvailable(MPIwCpp17)
