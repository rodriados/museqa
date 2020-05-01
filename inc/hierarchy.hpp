/**
 * Multiple Sequence Alignment hierarchy header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020 Rodrigo Siqueira
 */
#pragma once

#include <utility>

namespace msa
{
    /**
     * A node for a general hierarchical data structure.
     * @tparam T The hierarchical nodes' contents type.
     * @tparam R The hierarchical nodes' reference type.
     * @since 0.1.1
     */
    template <typename T, typename R = void>
    struct hierarchy
    {
        /**
         * The hierarchy's nodes' contents type.
         * @since 0.1.1
         */
        using element_type = T;

        /**
         * If no reference type is given, for connecting a node to its direct parent
         * and children, it's assumed an ordinary pointer will be used as usual.
         * @since 0.1.1
         */
        using reference_type = typename std::conditional<
                std::is_same<void, R>::value
            ,   hierarchy *
            ,   R
            >::type;

        static_assert(std::is_scalar<reference_type>::value, "hierarchy node reference must be scalar");

        /**
         * Definition of an undefined or unset node reference on the hierarchy.
         * In practice, this value will default to zero.
         * @since 0.1.1
         */
        static constexpr reference_type undefined = {};

        element_type self;                                  /// The node's self representation or contents.
        reference_type parent = undefined;                  /// The node's hierarchical parent reference.
        reference_type child[2] = {undefined, undefined};   /// The node's hierarchical children references.
    };
}
