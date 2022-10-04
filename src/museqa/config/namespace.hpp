/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Namespace configuration and macro definitions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

/**
 * Defines the namespace in which the library lives. This might be overriden if
 * the default namespace value is already in use.
 * @since 1.0
 */
#if defined(MUSEQA_OVERRIDE_NAMESPACE)
  #define MUSEQA_NAMESPACE MUSEQA_OVERRIDE_NAMESPACE
#else
  #define MUSEQA_NAMESPACE museqa
#endif

/**
 * This macro is used to open the `museqa::` namespace block and must not be in
 * any way overriden. This namespace must not be prefixed by any other namespaces
 * to avoid problems when allowing the use some of the library's facilities to with
 * STL's algorithms, structures or constructions.
 * @since 1.0
 */
#define MUSEQA_BEGIN_NAMESPACE   \
    namespace MUSEQA_NAMESPACE { \
        inline namespace v1 {    \
            namespace museqa = MUSEQA_NAMESPACE;

/**
 * This macro is used to close the `museqa::` namespace block and must not be in
 * any way overriden.
 * @since 1.0
 */
#define MUSEQA_END_NAMESPACE     \
    }}
