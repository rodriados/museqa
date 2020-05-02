/**
 * Multiple Sequence Alignment dendogram header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020 Rodrigo Siqueira
 */
#pragma once

#include <limits>
#include <cstdint>
#include <utility>

#include <buffer.hpp>
#include <allocator.hpp>
#include <hierarchy.hpp>

namespace msa
{
    namespace detail
    {
        namespace dendogram
        {
            /**
             * Represents a dendogram's node.
             * @tparam T The node's contents type.
             * @tparam D The node's distance type.
             * @tparam R The hierarchy reference type.
             * @since 0.1.1
             */
            template <typename T, typename D = double, typename R = size_t>
            struct node : public hierarchy<T, R>
            {
                static_assert(std::is_integral<R>::value, "dendogram node reference must be integral");
                static_assert(std::is_arithmetic<D>::value, "dendogram distance type must be arithmetic");

                using distance_type = D;

                static constexpr distance_type farthest = std::numeric_limits<distance_type>::max();

                distance_type distance = farthest;  /// The distance from this node to its parent.
                uint32_t level = 0;                 /// The node's level or height in dendogram.
            };
        }
    }

    /**
     * A dendogram is a special type of binary tree, where all non-leaf nodes must
     * have exactly two children nodes. Nodes are stored contiguously in memory.
     * @tparam T The dendogram's nodes' contents type.
     * @tparam D The dendogram's nodes' distance type.
     * @tparam R The type to reference dendogram's nodes.
     * @since 0.1.1
     */
    template <typename T, typename D = double, typename R = size_t>
    class dendogram : protected buffer<detail::dendogram::node<T, D, R>>
    {
        protected:
            using node_type = detail::dendogram::node<T, D, R>;
            using underlying_buffer = buffer<node_type>;

        public:
            using element_type = typename node_type::element_type;
            using distance_type = typename node_type::distance_type;
            using reference_type = typename node_type::reference_type;

        public:
            static constexpr reference_type undefined = node_type::undefined;

        protected:
            uint32_t m_leaves = 0;              /// The total number of leaves or points in dendogram.

        public:
            inline dendogram() noexcept = default;
            inline dendogram(const dendogram&) noexcept = default;
            inline dendogram(dendogram&&) noexcept = default;

            inline dendogram& operator=(const dendogram&) = default;
            inline dendogram& operator=(dendogram&&) = default;

            using underlying_buffer::operator[];

            /**
             * Retrieves the dendogram's root node.
             * @return The dendogram's root.
             */
            inline node_type& root()
            {
                return operator[](this->size() - 1);
            }

            /**
             * Retrieves the dendogram's const-qualified root node.
             * @return The const-qualified dendogram's root.
             */
            inline const node_type& root() const
            {
                return operator[](this->size() - 1);
            }

            /**
             * Informs the number of leaves in dendogram.
             * @return The total amount of leaf nodes in dendogram.
             */
            inline uint32_t leaves() const noexcept
            {
                return m_leaves;
            }

            /**
             * Creates a new dendogram with given number of points as leaves.
             * @param leaves The number of points in dendogram.
             * @return The newly created dendogram instance.
             */
            static inline dendogram make(uint32_t leaves) noexcept
            {
                return dendogram {underlying_buffer::make((leaves << 1) - 1), leaves};
            }

            /**
             * Creates a new dendogram with given number of points as leaves.
             * @param allocator The allocator to be used to create new dendogram.
             * @param leaves The number of points in dendogram.
             * @return The newly created dendogram instance.
             */
            static inline dendogram make(const msa::allocator& allocator, uint32_t leaves) noexcept
            {
                return dendogram {underlying_buffer::make(allocator, (leaves << 1) - 1), leaves};
            }

        protected:
            /**
             * Initializes a new dendogram from its underlying buffer.
             * @param raw The dendogram's underlying buffer.
             * @param leaves The number of points or leaves in dendogram.
             */
            inline dendogram(underlying_buffer&& raw, uint32_t leaves) noexcept
            :   underlying_buffer {std::forward<decltype(raw)>(raw)}
            ,   m_leaves {leaves}
            {}
    };
}
