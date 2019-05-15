/**
 * Multiple Sequence Alignment phylogeny module interface header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PHYLOGENY_CUH_INCLUDED
#define PHYLOGENY_CUH_INCLUDED

#include "phylogeny/tree.cuh"
#include "phylogeny/phylogeny.cuh"

/**
 * Represents the hierarchy of a pseudo-phylogenetic tree.
 * @since 0.1.1
 */
typedef phylogeny::Tree Tree;

/**
 * The phylogeny module manager.
 * @since 0.1.1
 */
typedef phylogeny::Phylogeny Phylogeny;

#endif