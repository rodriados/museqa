/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Namespace configuration and macro definitions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

/**
 * This macro is used to open the `museqa::` namespace block and must not be in
 * any way overriden. This namespace must not be prefixed by any other namespaces
 * to avoid problems when allowing the use some of the library's facilities to with
 * STL's algorithms, structures or constructions.
 * @since 1.0
 */
#define MUSEQA_BEGIN_NAMESPACE                                                  \
    namespace museqa {                                                          \
    inline namespace v1 {

/**
 * This macro is used to close the `museqa::` namespace block and must not be in
 * any way overriden.
 * @since 1.0
 */
#define MUSEQA_END_NAMESPACE                                                    \
    }}
