/**
 * Multiple Sequence Alignment neighbor-joining header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019-2020 Rodrigo Siqueira
 */
#pragma once

#include <limits>
#include <vector>

#include <buffer.hpp>
#include <pairwise.cuh>
#include <reflection.hpp>

#include <phylogeny/phylogeny.cuh>

namespace msa
{
    namespace phylogeny
    {
        namespace njoining
        {
            /**
             * Informs the maximum distance score possible.
             * @since 0.1.1
             */
            constexpr static const score infinity = std::numeric_limits<score>::has_infinity
                ? std::numeric_limits<score>::infinity()
                : std::numeric_limits<score>::max();

            /**
             * Holds the bare minimal information about a joinable candidate pair.
             * @since 0.1.1
             */
            struct candidate
            {
                oturef ref[2] = {undefined, undefined}; /// The OTU pair references.
                score distance = infinity;              /// The distance between the OTU pair.

                __host__ __device__ inline candidate() noexcept = default;
                __host__ __device__ inline candidate(const candidate&) noexcept = default;
                __host__ __device__ inline candidate(candidate&&) noexcept = default;

                /**
                 * Instantiates a new candidate pair from its internal values.
                 * @param x The first OTU reference value.
                 * @param y The second OTU reference value.
                 * @param distance The distance between the OTUs.
                 */
                __host__ __device__ inline candidate(oturef x, oturef y, score distance) noexcept
                :   ref {x, y}
                ,   distance {distance}
                {}

                __host__ __device__ inline candidate& operator=(const candidate&) noexcept = default;
                __host__ __device__ inline candidate& operator=(candidate&&) noexcept = default;

                /**
                 * Implements a volatile-qualified copy operator.
                 * @param other The volatile-qualified candidate to be copied.
                 * @return A volatile-qualified candidate instance.
                 */
                __host__ __device__ inline volatile candidate& operator=(volatile candidate& other) volatile
                {
                    ref[0] = other.ref[0];
                    ref[1] = other.ref[1];
                    distance = other.distance;
                    return *this;
                }
            };

            /**
             * Represents a joinable OTU pair candidate.
             * @since 0.1.1
             */
            struct joinable : public reflector
            {
                oturef ref[2] = {undefined, undefined}; /// The OTU pair references.
                score delta[2] = {infinity, infinity};  /// The selected OTU pair's deltas.
                score distance = infinity;              /// The distance between the OTU pair.

                __host__ __device__ inline joinable() noexcept = default;
                __host__ __device__ inline joinable(const joinable&) noexcept = default;
                __host__ __device__ inline joinable(joinable&&) noexcept = default;

                /**
                 * Creates a new joinable OTU pair from a candidate and its deltas.
                 * @param c The pair candidate to create instance from.
                 * @param dx The delta distance from the first OTU to its parent.
                 * @param dy The delta distance from the second OTU to its parent.
                 */
                __host__ __device__ inline joinable(const candidate& c, score dx, score dy) noexcept
                :   ref {c.ref[0], c.ref[1]}
                ,   delta {dx, dy}
                ,   distance {c.distance}
                {}

                __host__ __device__ inline joinable& operator=(const joinable&) noexcept = default;
                __host__ __device__ inline joinable& operator=(joinable&&) noexcept = default;

                #if !defined(__msa_compiler_nvcc)
                    using reflex = decltype(reflect(ref, delta, distance));
                #endif
            };

            /**
             * Represents a general neighbor-joining algorithm for solving the step
             * of building a phylogenetic tree.
             * @since 0.1.1
             */
            struct algorithm : public phylogeny::algorithm
            {
                virtual auto reduce(joinable&) const -> joinable;
                virtual auto run(const context&) const -> tree = 0;
            };

            /*
             * The list of all available neighbor-joining algorithm implementations.
             */
            extern auto hybrid_linear() -> phylogeny::algorithm *;
            extern auto hybrid_symmetric() -> phylogeny::algorithm *;
            extern auto sequential_linear() -> phylogeny::algorithm *;
            extern auto sequential_symmetric() -> phylogeny::algorithm *;
        }
    }
}
