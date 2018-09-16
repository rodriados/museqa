/**
 * Multiple Sequence Alignment pairwise interface header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef PAIRWISE_HPP_INCLUDED
#define PAIRWISE_HPP_INCLUDED

#pragma once

#include "pairwise/pairwise.hpp"

/*
 * Exposes pairwise classes to the global namespace.
 * @since 0.1.alpha
 */
typedef pairwise::Score Score;
typedef pairwise::Pairwise Pairwise;

#endif
