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

#if defined(__CUDACC__) && !defined(msa_compile_cuda)
  #define msa_compile_cuda 1
#endif

/**
 * Purifies the type to its base, removing all extents it might have.
 * @tparam T The type to have its base extracted.
 * @since 0.1.1
 */
template <typename T>
using Base = typename std::remove_extent<T>::type;

/**
 * Purifies an array type to its base.
 * @tparam T The type to be purified.
 * @since 0.1.1
 */
template <typename T>
using Pure = typename std::conditional<
        std::is_array<T>::value && !std::extent<T>::value
    ,   Base<T>
    ,   T
    >::type;

/**
 * Represents a function pointer type.
 * @tparam F The function signature type.
 * @since 0.1.1
 */
template <typename F>
using Functor = typename std::enable_if<std::is_function<F>::value, F*>::type;

#endif