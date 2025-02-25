/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Compiler-time macros encoding Museqa release version.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

/**
 * The preprocessor macros encoding the current Museqa library release version.
 * This is guaranteed to change with every Museqa release.
 */
#define MUSEQA_VERSION 10000

/**
 * The preprocessor macros encoding the release policy's values to the current Museqa
 * library release version.
 */
#define MUSEQA_VERSION_MAJOR (MUSEQA_VERSION / 10000)
#define MUSEQA_VERSION_MINOR (MUSEQA_VERSION / 100 % 100)
#define MUSEQA_VERSION_PATCH (MUSEQA_VERSION % 100)
/**#@-*/
