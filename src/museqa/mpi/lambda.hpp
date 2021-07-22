/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Operator functions for MPI reduce collective operations.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#if !defined(MUSEQA_AVOID_MPI)

#include <mpi.h>

#include <map>
#include <utility>

#include <museqa/mpi/type.hpp>
#include <museqa/mpi/common.hpp>
#include <museqa/utility.hpp>
#include <museqa/utility/functor.hpp>
#include <museqa/memory/pointer/shared.hpp>

namespace museqa
{
    namespace mpi
    {
        /**
         * Represents a user-defined MPI aggregation operator which can be directly
         * used with native MPI functions.
         * @since 1.0
         */
        class lambda;

        namespace function
        {
            /**
             * The type for a MPI operator instance. An identifier must exist for
             * every operator to be used with MPI collective operations.
             * @since 1.0
             */
            using id = MPI_Op;

            /**
             * The raw function type required by MPI to create a new operator for
             * its collective operations.
             * @since 1.0
             */
            using raw = void(void*, void*, int*, type::id*);
        }

        /**
         * Represents a function that can be used as an aggregator with MPI collective
         * operations, such as reduce, all-reduce and scan.
         * @since 1.0
         */
        class lambda : private memory::pointer::shared<void>
        {
          private:
            typedef function::id reference_type;
            typedef memory::pointer::shared<void> underlying_type;

          public:
            typedef function::raw *functor_type;    /// The native MPI operator function type.

          public:
            inline lambda() noexcept = default;
            inline lambda(const lambda&) noexcept = default;
            inline lambda(lambda&&) noexcept = default;

            /**
             * Creates a new MPI operator function for collective operations.
             * @param functor The user-defined function pointer.
             * @param commutative Is the operator commutative?
             */
            inline lambda(functor_type functor, bool commutative = true) noexcept(museqa::unsafe)
              : underlying_type {create(functor, commutative)}
            {}

            inline lambda& operator=(const lambda&) = default;
            inline lambda& operator=(lambda&&) = default;

            /**
             * Converts this object into a raw MPI operator reference, thus allowing
             * the user to seamlessly use the lambda with native MPI functions.
             * @return The raw MPI operator identifier.
             */
            inline operator reference_type() const noexcept
            {
                return (reference_type) this->m_ptr;
            }

          private:
            /**
             * Registers a new function as an operator within MPI's internal machinery.
             * @param functor The user-defined function pointer.
             * @param commutative Is the operator commutative?
             * @return The internal pointer instance.
             */
            inline auto create(functor_type functor, bool commutative = true) noexcept(museqa::unsafe)
            -> underlying_type
            {
                mpi::check(MPI_Op_create(functor, commutative, (reference_type*) &this->m_ptr));
                auto destructor = [](void *ptr) { mpi::check(MPI_Op_free((reference_type*) &ptr)); };
                return underlying_type {this->m_ptr, destructor};
            }
        };

        namespace function
        {
            /**
             * Informs the currently active MPI operator function. This is necessary
             * to recover a wrapped function from within MPI execution.
             * @since 1.0
             */
            extern function::id active;

            /**
             * Maps a function identifier to its actual user-defined implementation.
             * Unfortunately, this is maybe the only reliable way to inject a wrapped
             * function into the operator actually called by MPI.
             * @since 1.0
             */
            extern std::map<function::id, void*> fmapper;

            /**
             * Wraps a typed function transforming it into a generic MPI operator.
             * @tparam T The type the operator works onto.
             * @param a The operation's first operand reference.
             * @param b The operation's second operand and output values reference.
             * @param count The total number of elements to process during execution.
             */
            template <typename T>
            void wrapper(void *a, void *b, int *count, type::id *)
            {
                using function_type = T(*)(const T&, const T&);
                auto f = reinterpret_cast<function_type>(fmapper[active]);

                for (int i = 0; i < *count; ++i)
                    static_cast<T*>(b)[i] = f(static_cast<T*>(a)[i], static_cast<T*>(b)[i]);
            }

            /**#@+
             * Declaration of built-in MPI operators. The use of these is highly
             * recommended and preferred if possible.
             * @since 1.0
             */
            static constexpr function::id const max     = MPI_MAX;
            static constexpr function::id const min     = MPI_MIN;
            static constexpr function::id const add     = MPI_SUM;
            static constexpr function::id const mul     = MPI_PROD;
            static constexpr function::id const andl    = MPI_LAND;
            static constexpr function::id const andb    = MPI_BAND;
            static constexpr function::id const orl     = MPI_LOR;
            static constexpr function::id const orb     = MPI_BOR;
            static constexpr function::id const xorl    = MPI_LXOR;
            static constexpr function::id const xorb    = MPI_BXOR;
            static constexpr function::id const minloc  = MPI_MINLOC;
            static constexpr function::id const maxloc  = MPI_MAXLOC;
            static constexpr function::id const replace = MPI_REPLACE;
            /**#@-*/
        }
    }

    namespace factory
    {
        namespace mpi
        {
            /**
             * Registers a user-defined function as a MPI operator.
             * @tparam T The operator's target operand type.
             * @param functor The function to wrap as a MPI operator.
             * @param commutative Is the operator commutative?
             * @return The registered MPI operator lambda.
             */
            template <typename T>
            inline auto lambda(
                T (*functor)(const T&, const T&)
              , bool commutative = true
            ) noexcept(museqa::unsafe) -> museqa::mpi::lambda
            {
                namespace mpi = museqa::mpi;
                auto lambda = mpi::lambda {mpi::function::wrapper<T>, commutative};
                mpi::function::fmapper[lambda] = reinterpret_cast<void*>(functor);
                return lambda;
            }

            /**
             * Registers a user-defined functor as a MPI operator.
             * @tparam T The operator's target operand type.
             * @param functor The functor instance to wrap as a MPI operator.
             * @param commutative Is the operator commutative?
             * @return The registered MPI operator lambda.
             */
            template <typename T>
            inline auto lambda(
                const utility::functor<T(const T&, const T&)>& functor
              , bool commutative = true
            ) noexcept(museqa::unsafe) -> museqa::mpi::lambda
            {
                return factory::mpi::lambda(*functor, commutative);
            }
        }
    }
}

#endif
