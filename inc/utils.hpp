/** 
 * Multiple Sequence Alignment utilities header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef UTILS_HPP_INCLUDED
#define UTILS_HPP_INCLUDED

#include <cstddef>
#include <utility>

/*
 * Creation of conditional macros that allow CUDA declarations to be used
 * seamlessly throughout the code without any problems.
 */
#ifdef __CUDA_ARCH__
  #define cudadecl __host__ __device__
#else
  #define cudadecl
#endif

/**
 * Purifies an array type to its base.
 * @tparam T The type to be purified.
 * @since 0.1.1
 */
template <typename T>
using Pure = typename std::remove_extent<T>::type;

/**
 * A universal pointer type. All pointer objects should be convertible to it.
 * @tparam T The pointer type.
 * @since 0.1.1
 */
template <typename T>
using Pointer = Pure<T> *;

/**
 * Represents a function carrying type.
 * @tparam R The function return type.
 * @tparam P The function parameter types.
 * @since 0.1.1
 */
template <typename R, typename... P>
using Functor = R (*)(P...);

#endif