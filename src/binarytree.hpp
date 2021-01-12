/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements a generic binary-tree data structure.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>
#include <utility>

#include "hierarchy.hpp"

namespace museqa
{
    /**
     * Represents a generic binary tree. This tree does not enforce opinions on
     * how its nodes are stored or iterated through.
     * @tparam T The tree's nodes' contents type.
     * @tparam R The tree's nodes' reference type.
     * @since 0.1.1
     */
    template <typename T, typename R = void>
    class binarytree
    {
        public:
            struct node;                        /// The binary tree's nodes' type.

        public:
            using element_type = typename node::element_type;
            using reference_type = typename node::reference_type;

        public:
            static constexpr reference_type undefined = node::undefined;

        protected:
            reference_type m_root = undefined;  /// The tree's root node reference.

        public:
            inline binarytree() noexcept = default;
            inline binarytree(const binarytree&) noexcept = default;
            inline binarytree(binarytree&&) noexcept = default;

            /**
             * Creates a new binary tree by acquiring a root node reference.
             * @param root The tree's root node's reference.
             */
            inline binarytree(reference_type root) noexcept
            :   m_root {root}
            {}

            virtual ~binarytree() = default;

            inline binarytree& operator=(const binarytree&) noexcept = default;
            inline binarytree& operator=(binarytree&&) noexcept = default;

            /**
             * Gives access to the tree's root node reference.
             * @return The tree's root node reference.
             */
            inline reference_type root() const noexcept
            {
                return m_root;
            }
    };

    namespace detail
    {
        namespace binarytree
        {
            /**#@+
             * Represents a generic binary tree node.
             * @tparam T The node's contents type.
             * @tparam R The node hierarchy reference type.
             * @since 0.1.1
             */
            template <typename T, typename R, typename = void>
            struct node : hierarchy<R>
            {
                T contents;             /// The node's contents.
            };

            template <typename T, typename R>
            struct node<T, R, typename std::enable_if<std::is_class<T>::value>::type> : T, hierarchy<R>
            {};
            /**#@-*/
        }
    }

    /**
     * The binary tree's generic node type.
     * @tparam T The tree's nodes' contents type.
     * @tparam R The tree's nodes' reference type.
     * @since 0.1.1
     */
    template <typename T, typename R>
    struct binarytree<T, R>::node : public detail::binarytree::node<T, R>
    {
        using element_type = T;         /// The binary tree's nodes' element type.
    };
}
