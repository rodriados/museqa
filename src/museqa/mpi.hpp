/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Imports the whole codebase around MPI, such wrappers and helper functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#if !defined(MUSEQA_AVOID_MPI)

#include <museqa/mpi/common.hpp>
#include <museqa/mpi/type.hpp>
#include <museqa/mpi/status.hpp>
#include <museqa/mpi/lambda.hpp>
#include <museqa/mpi/communicator.hpp>
#include <museqa/mpi/collective.hpp>

namespace museqa
{
    namespace mpi
    {
        /**
         * Initializes MPI state and system machinery and automatically frees all
         * the used resources as soon as the system instance is destroyed. Thus,
         * an instance must be created before calling any MPI function and must
         * be destroyed only after all MPI operations are done. Therefore, it is
         * recommended to create an static instance at the top of main.
         * @since 1.0
         */
        struct state
        {
            inline state() noexcept = delete;
            inline state(const state&) noexcept = delete;
            inline state(state&&) noexcept = delete;

            /**
             * Initializes the MPI machinery.
             * @param argc The number of arguments received via command-line.
             * @param argv The program's command-line arguments.
             */
            inline state(int& argc, char**& argv)
            {
                mpi::init(argc, argv);
            }

            /**
             * Destroys all MPI state and finalize its internal machinery. No MPI
             * functions are allowed to be called after MPI state is destroyed.
             * @see museqa::mpi::finalize
             */
            inline ~state()
            {
                mpi::finalize();
            }

            inline state& operator=(const state&) noexcept = delete;
            inline state& operator=(state&&) noexcept = delete;
        };
    }
}

#endif
