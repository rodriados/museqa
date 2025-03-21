# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file A collection of miscellaneous helper CMake functions.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2025-present Rodrigo Siqueira

# Retrieves the current project's version from the given source file.
# @param var The name of the variable to search for and return version to.
# @param filename The source file to retrieve project version from.
function(get_project_version var filename)
  file(READ "${filename}" contents)
  if(NOT contents MATCHES "${var} ([0-9]+)([0-9][0-9])([0-9][0-9])")
    message(FATAL_ERROR "Cannot find Museqa version.")
  endif()

  math(EXPR VERSION_MAJOR ${CMAKE_MATCH_1})
  math(EXPR VERSION_MINOR ${CMAKE_MATCH_2})
  math(EXPR VERSION_PATCH ${CMAKE_MATCH_3})

  set(COMPLETE_VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
  set(${var} "${COMPLETE_VERSION}" PARENT_SCOPE)
endfunction()
