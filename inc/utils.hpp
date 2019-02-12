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
#if !defined(__host__) && !defined(__device__)
  #define __host__
  #define __device__
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
 * Represents a function pointer type.
 * @tparam F The function signature type.
 * @since 0.1.1
 */
template <typename F>
using Functor = typename std::enable_if<std::is_function<F>::value, F*>::type;

#endif