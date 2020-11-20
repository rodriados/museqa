/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements an interface for pipelines of modules.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#pragma once

#include <utility>

#include "io.hpp"
#include "tuple.hpp"
#include "utils.hpp"
#include "pointer.hpp"
#include "exception.hpp"

namespace museqa
{
    namespace pipeline
    {
        /**
         * A conduit carries all relevant information from a module into the next.
         * Any data transmission via two consecutive modules should only happen
         * via a conduit. This struct shall be specialized for each module.
         * @since 0.1.1
         */
        struct conduit
        {
            inline explicit conduit() noexcept = default;
            inline explicit conduit(const conduit&) = default;
            inline explicit conduit(conduit&&) = default;

            inline virtual ~conduit() noexcept = default;

            inline conduit& operator=(const conduit&) = default;
            inline conduit& operator=(conduit&&) = default;
        };

        /**
         * The base of a pipeline module. Modules can be chained in a pipeline so
         * they are executed sequentially, one after the other. All modules in a
         * pipeline must inherit from this struct. Also, they all must indicate
         * which module they expect to be its previous. To run, a module must implement
         * the `run` function, which will always take the command line manager instance
         * and the previous module's conduit.
         * @since 0.1.1
         */
        struct module
        {
            using previous = void;                  /// Indicates the expected previous module.
            using conduit = pipeline::conduit;      /// The module's conduit type.

            virtual auto check(const io::service&) const -> bool = 0;
            virtual auto run(const io::service&, const pointer<conduit>&) const -> pointer<conduit> = 0;
            virtual auto name() const -> const char * = 0;
        };

        /**
         * Converts an unknown conduit reference to that of the expected conduit
         * type of a module. This function checks whether the conversion is possible.
         * @tparam T The type of module receiving the conduit to be converted.
         * @param target The instance to be converted into the expected type.
         * @return The converted instance reference.
         */
        template <typename T>
        inline auto convert(const conduit& target) noexcept
        -> typename std::enable_if<
                std::is_base_of<module, T>::value &&
                std::is_base_of<conduit, typename T::conduit>::value
            ,   const typename T::conduit&
            >::type
        {
            return dynamic_cast<const typename T::conduit&>(target);
        }
    }

    namespace detail
    {
        namespace pipeline
        {
            /**#@+
             * Auxiliary funciton for checking whether the pipelined modules are
             * chainable. To achieve such task, we look at each module's previous
             * type, as they should match the previous module in the pipeline.
             * @tparam P The previously analyzed module.
             * @tparam T The current module being analyzed.
             * @return Are the pipelined modules chainable?
             */
            template <typename T>
            constexpr auto chainable() -> bool
            {
                return true;
            }

            template <typename P, typename T, typename ...U>
            constexpr auto chainable() -> bool
            {
                using previous = typename T::previous;

                return (std::is_same<previous, P>::value || std::is_base_of<previous, P>::value)
                    && std::is_base_of<museqa::pipeline::module, T>::value
                    && chainable<T, U...>();
            }
            /**#@-*/
        }
    }

    namespace pipeline
    {
        /**
         * Manages the pipelined modules execution. From the given list of pipelined
         * modules, verify whether they can be actually chained and run them.
         * @tparam T The list of modules to be chained.
         * @since 0.1.1
         */
        template <typename ...T>
        class runner
        {
            static_assert(utils::all(std::is_base_of<module, T>()...), "pipeline can only handle modules");
            static_assert(utils::all(std::is_default_constructible<T>()...), "modules must default construct");
            static_assert(detail::pipeline::chainable<void, T...>(), "given modules cannot be chained");

            public:
                static constexpr size_t count = sizeof...(T);   /// The number of chained modules.

            protected:
                using module_tuple = tuple<T...>;               /// The tuple of chained modules types.
                using pipe = pointer<pipeline::conduit>;        /// The return type expected from modules.

            public:
                /**
                 * Runs the pipeline and returns the last module's result.
                 * @param io The pipeline's IO service instance.
                 * @return The last module's resulting value.
                 */
                inline auto run(const io::service& io) const -> pipe
                {
                    const module *ptr[count];
                    const module_tuple modules = {};

                    auto extract = [](const module& m) { return &m; };
                    utils::tie(ptr) = utils::apply(extract, modules);

                    if(!verify(ptr, io))
                        throw exception {"pipeline verification failed"};

                    return execute(ptr, io);
                }

            protected:
                /**
                 * Verifies whether all modules will be in a valid state given the
                 * pipeline's command line arguments.
                 * @param modules The list of pipeline's modules instances.
                 * @param io The pipeline's IO service instance.
                 * @return Are all modules in a valid state?
                 */
                inline virtual bool verify(const module *modules[], const io::service& io) const
                {
                    for(size_t i = 0; i < count; ++i)
                        if(!modules[i]->check(io))
                            return false;

                    return true;
                }

                /**
                 * Executes the pipeline's module in sequence.
                 * @param modules The list of pipeline's modules instances.
                 * @param io The pipeline's IO service instance.
                 * @return The pipeline's final module's result.
                 */
                inline virtual pipe execute(const module *modules[], const io::service& io) const
                {
                    auto previous = pipe {};

                    for(size_t i = 0; i < count; ++i)
                        previous = std::move(modules[i]->run(io, previous));

                    return previous;
                }
        };
    }
}
