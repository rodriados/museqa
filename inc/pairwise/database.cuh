/**
 * Multiple Sequence Alignment pairwise database header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PW_DATABASE_CUH_INCLUDED
#define PW_DATABASE_CUH_INCLUDED

#include <cuda.cuh>
#include <buffer.hpp>
#include <pointer.hpp>
#include <database.hpp>
#include <sequence.hpp>
#include <exception.hpp>

namespace pairwise
{
    /**
     * Stores a list of sequences from a database as a single contiguous sequence.
     * No description nor any other metadata will be stored alongside the sequences,
     * which will all solely be identified by their indeces. These indeces will
     * be kept the same as in the original database.
     * @since 0.1.1
     */
    class database : public sequence
    {
        public:
            using element_type = sequence_view;             /// The database's element type.

        protected:
            using underlying_type = sequence;               /// The database's underlying type.
            using entry_buffer = buffer<element_type>;      /// The database's underlying buffer type.

        protected:
            entry_buffer mviews;                            /// The buffer of sequence views.

        public:
            inline database() noexcept = default;
            inline database(const database&) noexcept = default;
            inline database(database&&) noexcept = default;

            /**
             * Initializes a contiguous database from a common database instance.
             * @param db The database to be transformed.
             */
            inline database(const ::database& db) noexcept
            :   underlying_type {merge(db)}
            ,   mviews {init(*this, db)}
            {}

            inline database& operator=(const database&) = default;
            inline database& operator=(database&&) = default;

            /**
             * Gives access to a specific sequence in the database.
             * @param offset The offset of requested sequence.
             * @return The requested sequence.
             */
            __host__ __device__ inline auto operator[](ptrdiff_t offset) const -> const element_type&
            {
                enforce(offset >= 0 && size_t(offset) < count(), "database index out of range");
                return mviews[offset];
            }

            /**
             * Informs the number of sequences in database.
             * @return The database's number of sequences.
             */
            __host__ __device__ inline size_t count() const
            {
                return mviews.size();
            }

            auto to_device() const -> database;

        private:
            /**
             * Initializes a new database from already compressed sequences and
             * their respective views.
             * @param blocks The encoded blocks of compressed database sequences.
             * @param views The sequence views upon the compressed sequences.
             */
            inline database(const underlying_type& blocks, const entry_buffer& views)
            :   underlying_type {blocks}
            ,   mviews {views}
            {}

            static auto init(underlying_type&, const ::database&) -> entry_buffer;
            static auto merge(const ::database&) -> underlying_type;
    };
}

#endif