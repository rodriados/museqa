/**
 * Multiple Sequence Alignment phylogeny module interface header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PHYLOGENY_CUH_INCLUDED
#define PHYLOGENY_CUH_INCLUDED

#include <phylogeny/phylogeny.cuh>

/**
 * Represents a pseudo-phylogenetic tree, so the sequences can be organized in a
 * hierarchical structure for them to be progressively aligned.
 * @since 0.1.1
 */
typedef phylogeny::tree tree;

#endif