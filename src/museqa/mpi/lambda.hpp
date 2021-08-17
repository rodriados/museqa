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

#include <museqa/utility.hpp>
#include <museqa/utility/functor.hpp>
#include <museqa/memory/pointer/shared.hpp>
#include <museqa/memory/pointer/weak.hpp>

#include <museqa/mpi/type.hpp>
#include <museqa/mpi/common.hpp>

namespace museqa
{
    namespace mpi
    {
        /**
         * Represents a function that can be used as an aggregator with MPI collective
         * operations, such as reduce, all-reduce and scan.
         * @since 1.0
         */
        class lambda : private memory::pointer::shared<void>
        {
          private:
            typedef MPI_Op reference_type;
            typedef memory::pointer::shared<void> underlying_type;

          public:
            typedef reference_type id;
            using functor_type = void (*)(void*, void*, int*, type::id*);

          public:
            inline lambda() noexcept = default;
            inline lambda(const lambda&) noexcept = default;
            inline lambda(lambda&&) noexcept = default;

            /**
             * Creates a new non-owning lambda from an already existing operator.
             * @param ref The operator to be wrapped into a lambda.
             */
            inline lambda(reference_type ref) noexcept
              : underlying_type {memory::pointer::weak<void>{ref}}
            {}

            /**
             * Creates a new MPI operator function for collective operations.
             * @param functor The user-defined function pointer.
             * @param commutative Is the operator commutative?
             */
            inline lambda(functor_type functor, bool commutative = true) noexcept(!safe)
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
            inline auto create(functor_type functor, bool commutative = true) noexcept(!safe)
            -> underlying_type
            {
                mpi::check(MPI_Op_create(functor, commutative, (reference_type*) &this->m_ptr));
                auto destructor = [](void *ptr) { mpi::check(MPI_Op_free((reference_type*) &ptr)); };
                return underlying_type {this->m_ptr, destructor};
            }

          public:
            static constexpr reference_type const max     = MPI_MAX;
            static constexpr reference_type const min     = MPI_MIN;
            static constexpr reference_type const add     = MPI_SUM;
            static constexpr reference_type const mul     = MPI_PROD;
            static constexpr reference_type const andl    = MPI_LAND;
            static constexpr reference_type const andb    = MPI_BAND;
            static constexpr reference_type const orl     = MPI_LOR;
            static constexpr reference_type const orb     = MPI_BOR;
            static constexpr reference_type const xorl    = MPI_LXOR;
            static constexpr reference_type const xorb    = MPI_BXOR;
            static constexpr reference_type const minloc  = MPI_MINLOC;
            static constexpr reference_type const maxloc  = MPI_MAXLOC;
            static constexpr reference_type const replace = MPI_REPLACE;
        };

        namespace impl
        {
            namespace lambda
            {
                /**
                 * Informs the currently active MPI operator function. This is necessary
                 * to recover a wrapped function from within MPI execution.
                 * @since 1.0
                 */
                extern mpi::lambda::id active;

                /**
                 * Maps a function identifier to its actual user-defined implementation.
                 * Unfortunately, this is maybe the only reliable way to inject
                 * a wrapped function into the operator actually called by MPI.
                 * @since 1.0
                 */
                extern std::map<mpi::lambda::id, void*> fmapper;

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
                    using function_type = T(const T&, const T&);
                    auto f = reinterpret_cast<function_type*>(fmapper[active]);

                    for (int i = 0; i < *count; ++i)
                        static_cast<T*>(b)[i] = (f)(static_cast<T*>(a)[i], static_cast<T*>(b)[i]);
                }
            }
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
            ) noexcept(!safe) -> museqa::mpi::lambda
            {
                namespace mpi = museqa::mpi;
                auto lambda = mpi::lambda {mpi::impl::lambda::wrapper<T>, commutative};
                mpi::impl::lambda::fmapper[lambda] = reinterpret_cast<void*>(functor);
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
            ) noexcept(!safe) -> museqa::mpi::lambda
            {
                return factory::mpi::lambda(*functor, commutative);
            }
        }
    }
}

#endif
