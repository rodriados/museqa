/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements generic CUDA-compatible unique pointers.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/utility.hpp>
#include <museqa/memory/pointer/common.hpp>
#include <museqa/memory/pointer/shared.hpp>

namespace museqa
{
    namespace memory
    {
        namespace pointer
        {
            /**
             * Implements an unique pointer, so that the encapsulated pointer can
             * only have one single owner during its lifetime.
             * @tparam T The type of pointer to be encapsulated.
             * @since 1.0
             */
            template <typename T>
            class unique : public shared<T>
            {
              private:
                typedef shared<T> underlying_type;

              public:
                using typename underlying_type::allocator_type;
                using typename underlying_type::element_type;

              public:
                __host__ __device__ inline constexpr unique() noexcept = default;
                __host__ __device__ inline constexpr unique(const unique&) noexcept = delete;
                __host__ __device__ inline constexpr unique(unique&&) noexcept = default;

                /**
                 * Builds a new unique pointer instance from a raw pointer.
                 * @param ptr The pointer to be encapsulated.
                 */
                __host__ __device__ inline explicit unique(element_type *ptr) noexcept
                  : underlying_type {ptr}
                {}

                /**
                 * Builds a new instance from a raw pointer and its allocator.
                 * @param ptr The pointer to be encapsulated.
                 * @param allocator The allocator of given pointer.
                 */
                inline unique(element_type *ptr, const allocator_type& allocator) noexcept
                  : underlying_type {ptr, allocator}
                {}

                /**
                 * The move constructor from a foreign pointer type.
                 * @tparam U The foreign pointer type to be moved.
                 * @param other The foreign instance to be moved.
                 */
                template <typename U>
                __host__ __device__ inline unique(unique<U>&& other) noexcept
                  : underlying_type {std::forward<decltype(other)>(other)}
                {}

                __host__ __device__ inline unique& operator=(const unique&) = delete;
                __host__ __device__ inline unique& operator=(unique&&) = default;

                /**
                 * The move-assignment operator from a foreign pointer type.
                 * @tparam U The foreign pointer type to be moved.
                 * @param other The foreign instance to be moved.
                 * @return This pointer object.
                 */
                template <typename U>
                __host__ __device__ inline unique& operator=(unique<U>&& other)
                {
                    return underlying_type::operator=(std::forward<decltype(other)>(other));
                }

              protected:
                using underlying_type::offset;
                using underlying_type::swap;
            };
        }
    }
}
