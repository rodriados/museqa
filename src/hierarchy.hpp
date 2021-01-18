/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements a base for a generic hierarchically binary data structure.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>
#include <utility>

namespace museqa
{
    /**
     * A generic hierarchically binary data structure node.
     * @tparam R The hierarchical nodes' reference type.
     * @since 0.1.1
     */
    template <typename R = void>
    struct hierarchy
    {
        /**
         * If no reference type is given for connecting a node to its direct parent
         * and children, it's assumed an ordinary pointer will be used as usual.
         * @since 0.1.1
         */
        using reference_type = typename std::conditional<
                std::is_void<R>::value
            ,   hierarchy *
            ,   R
            >::type;

        static_assert(std::is_scalar<reference_type>::value, "hierarchy node reference must be scalar");

        /**
         * Definition of an undefined or unset node reference on the hierarchy.
         * In practice, this value will default to zero or a virtual infinity.
         * @since 0.1.1
         */
        static constexpr reference_type undefined = (reference_type) typename std::conditional<
                !std::is_pointer<reference_type>::value
            ,   std::integral_constant<int8_t, ~0x00>
            ,   nullptr_t
            >::type {};

        reference_type parent = undefined;                  /// The node's hierarchical parent reference.
        reference_type child[2] = {undefined, undefined};   /// The node's hierarchical children references.
    };
}
