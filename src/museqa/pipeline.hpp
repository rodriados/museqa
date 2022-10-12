/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Pipeline for modules of heuristic algorithms.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <map>
#include <string>
#include <utility>

#include <museqa/environment.h>
#include <museqa/utility.hpp>

#include <museqa/memory/pointer.hpp>
#include <museqa/utility/tuple.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace pipeline
{
    /**
     * A pipe connects two modules which are executed subsequently within a pipeline,
     * thus carrying a resulting object from one module to another. The destination
     * module is responsible for interpreting and converting the pipe's contents
     * into an usable type.
     * @since 1.0
     */
    using pipe = memory::pointer::shared<void>;

    /**
     * The abstract base of a pipeline module. Modules can be chained in a pipeline
     * so they are executed sequentially and their respective results are carried
     * through the chain. All modules within a pipeline are required to implement
     * this struct. It is the module's responsibility to interpret the data being
     * transmited through the pipe.
     * @since 1.0
     */
    struct module
    {
        /**
         * Initializes the module before the pipeline execution. Within a pipeline,
         * the modules are initialized in the same order they are executed.
         * @param p The pipeline's transitive pipe instance.
         */
        virtual void initialize(pipe& p) const { }

        /**
         * Finalizes the module after the pipeline execution. Within a pipeline,
         * the modules are finalized in the opposite order they are executed.
         * @param p The pipeline's transitive pipe instance.
         */
        virtual void finalize(pipe& p) const { }

        /**
         * Executes the module logic. After a valid pipeline initialization, the
         * module's main logic must be implemented by this method.
         * @param p The pipeline's transitive pipe instance.
         */
        virtual void run(pipe& p) const = 0;

        /**
         * Retrieves the module's identifying name. Each module must have a distinct
         * name for identification purposes within reports.
         * @return The module's name.
         */
        virtual auto name() const -> std::string = 0;
    };

    /**
     * The abstract base of a module middleware. A middleware allows a module to
     * have its functionality easily extended. The middleware may override any of
     * the wrapped module's methods, and must explicitly bubble up or avoid the
     * overriden method's logic by explicitly referring to the wrapped module.
     * @tparam M The module type wrapped by the middleware.
     * @since 1.0
     */
    template <typename M>
    struct middleware : public std::enable_if<std::is_base_of<module, M>::value, M>::type
    {
        /**
         * Executes the middleware logic. The middleware is responsible for explicitly
         * bubbling up the module's execution or short-circuiting it.
         * @param p The pipeline's transitive pipe instance.
         */
        inline virtual void run(pipe& p) const override
        {
            M::run(p);
        }
    };

    namespace detail
    {
        /**#@+
         * Automatically composes a list of middlewares around a target module,
         * allowing its functionality to be seamlessly extended. The resulting type
         * of this composition is transparently also a module.
         * @tparam M The base module to be extended by a middleware.
         * @tparam T The list of middlewares to extend the module with.
         * @since 1.0
         */
        template <typename M, template <class> class ...T>
        struct autowire;

        template <typename M>
        struct autowire<M> : public identity<M> {};

        template <typename M, template <class> class T, template <class> class ...U>
        struct autowire<M, T, U...> : public std::enable_if<
            std::is_base_of<middleware<M>, T<M>>::value
          , identity<T<typename detail::autowire<M, U...>::type>>
        >::type {};
        /**#@-*/
    }

    /**
     * Automatically creates a composition of middlewares around a base module,
     * thus allowing the module's functionality to be easily extended.
     * @tparam M The module to be extended by a middleware.
     * @tparam T The list of middlewares to extend the module with.
     * @since 1.0
     */
    template <typename M, template <class> class ...T>
    using autowire = typename detail::autowire<M, T...>::type;

    /**
     * Manages the execution of a pipeline of modules and middlewares.
     * @tparam M The list of modules to be executed.
     * @since 1.0
     */
    template <typename ...M>
    class runner
    {
        public:
            inline static constexpr size_t count = sizeof...(M);

        static_assert(utility::all(std::is_base_of<module, M>()...), "pipeline can only run modules");
        static_assert(utility::all(std::is_default_constructible<M>()...), "modules must be default constructible");

        public:
            /**
             * Executes the pipelined modules and transitively bubbles a pipe instance
             * through the executing modules and steps.
             * @param pipe The input pipe to transit to the first module in line.
             * @return The resulting pipe instance.
             */
            inline static auto run(const pipeline::pipe& pipe = {}) -> pipeline::pipe
            {
                const module* modules[count];
                const auto mtuple = utility::tuple(M()...);

                auto result = pipeline::pipe(pipe);
                auto f = [](const module& m) { return &m; };
                utility::tie(modules) = utility::apply(f, mtuple);

                // Invoking each module's initialization routine. When initializing,
                // a module should not execute any compute-heavy operation but only
                // perform validations and initialize the most basic objects it
                // needs to execute correctly.
                for (size_t i = 0; i < count; ++i)
                    modules[i]->initialize(result);

                // Invoking the module's core logic. There is no actual limit to
                // what a module can do, though it is recommended that its execution
                // is guaranteed to finish in polynomial time and that it is isolated
                // from other modules on the pipeline.
                for (size_t i = 0; i < count; ++i)
                    modules[i]->run(result);

                // Invoking the module's finalization rountine. When finalizing,
                // a module should not execute any compute-heavy operation but only
                // perform clean up and result storage operations.
                for (size_t i = 1; i <= count; ++i)
                    modules[count-i]->finalize(result);

                return result;
            }
    };
}

MUSEQA_END_NAMESPACE
