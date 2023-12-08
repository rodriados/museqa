/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Pipeline for modules of heuristic algorithms.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <utility>
#include <unordered_map>

#include <museqa/environment.h>
#include <museqa/utility.hpp>

#include <museqa/memory/pointer.hpp>
#include <museqa/utility/tuple.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace pipeline
{
    /**
     * The state of a pipeline is a generic container that carries stateful data
     * between different modules and middlewares during the execution of the pipeline.
     * @since 1.0
     */
    class state_t
    {
        private:
            typedef memory::pointer::shared_t<void> entry_t;
            typedef std::unordered_map<std::string, entry_t> context_t;

        private:
            context_t m_context = {};

        public:
            inline explicit state_t() = default;
            inline explicit state_t(const state_t&) = default;
            inline explicit state_t(state_t&&) = default;

            inline state_t& operator=(const state_t&) = default;
            inline state_t& operator=(state_t&&) = default;

            /**
             * Retrieves data referenced by the given key from the context.
             * @tparam T The type of data to retrieve.
             * @param key The key to be requested from or created within the context.
             * @return The data retrieved from the context.
             */
            template <typename T = void>
            inline memory::pointer::shared_t<T>& get(const std::string& key)
            {
                return reinterpret_cast<memory::pointer::shared_t<T>&>(m_context[key]);
            }

            /**
             * Sets a new key or re-assigns a key with the given value in the context.
             * @tparam T The type of data to be assigned to key.
             * @param key The key to be assigned within the context.
             * @param value The value to assign to the key in the context.
             */
            template <typename T = void>
            inline void set(const std::string& key, memory::pointer::shared_t<T>& value)
            {
                m_context.insert_or_assign(key, reinterpret_cast<entry_t&>(value));
            }

            /**
             * Removes data referenced by a given key from the context. No errors
             * or exceptions will be raised if key is not found in the context.
             * @param key The key to be removed from the context.
             * @return Has the key been found and removed?
             */
            inline bool remove(const std::string& key)
            {
                return (bool) m_context.erase(key);
            }
    };

    /**
     * The pipe connector responsible for carrying stateful data between modules
     * and middlewares down the execution chain of a pipeline.
     * @since 1.0
     */
    using pipe_t = memory::pointer::shared_t<state_t>;

    /**
     * The abstract base of a pipeline module. Modules are chained in a pipeline
     * and executed sequentially. Their respective results are carried down the
     * chain with a pipe. All modules within a pipeline are required to inherit
     * the abstract module base. It is the module's responsibility to correctly
     * interpret the data being transmitted through the pipe.
     * @since 1.0
     */
    struct module_t
    {
        /**
         * The method to run before the module's core logic. If needed, a module
         * can perform validations or initialize state before before the core logic.
         * @param pipe The pipeline's transitive state instance.
         */
        virtual void before(pipe_t& pipe) const { }

        /**
         * The method to execute the module's core logic. After the pipeline has
         * been successfully initialized, the module's core logic is can be run.
         * @param pipe The pipeline's transitive state instance.
         */
        virtual void run(pipe_t& pipe) const = 0;

        /**
         * The method to run after the module's core logic. Within a pipeline, modules
         * have this method called in the opposite order to which they are defined.
         * @param pipe The pipeline's transitive state instance.
         */
        virtual void after(pipe_t& pipe) const { }
    };

    /**
     * The base of a middleware. A middleware allows modules to be wrapped in groups
     * and have their functionalities exteded. A middleware may override any of
     * the wrapped module's methods, and can explicitly bubble up or avoid the original
     * module's behavior by referring to the wrapped module.
     * @tparam M The list of module types wrapped by the middleware.
     * @since 1.0
     */
    template <typename ...M>
    class middleware_t : public pipeline::module_t
    {
        public:
            typedef utility::tuple_t<M...> sequence_t;

        protected:
            const sequence_t m_sequence = {};

        static_assert(
            utility::all(std::is_base_of<pipeline::module_t, pure_t<M>>::value...)
          , "every pipeline step must ultimately inherit from a module"
        );

        public:
            inline constexpr middleware_t() = default;
            inline constexpr middleware_t(const middleware_t&) = default;
            inline constexpr middleware_t(middleware_t&&) = delete;

            /**
             * Configures a new pipeline middleware with the modules it must execute.
             * @param modules The middleware's modules instance list.
             */
            inline constexpr middleware_t(M&&... modules)
              : m_sequence (std::forward<decltype(modules)>(modules)...)
            {}

            inline middleware_t& operator=(const middleware_t&) = delete;
            inline middleware_t& operator=(middleware_t&&) = delete;

            /**
             * Executes the validation or initialization logic of all modules referenced
             * by the middleware. Note that they are guaranteed to be called in
             * the order in which they are declared.
             * @param pipe The pipeline's transitive state instance.
             */
            inline virtual void before(pipe_t& pipe) const override
            {
                utility::foreach(&module_t::before, m_sequence, pipe);
            }

            /**
             * Executes the core logic of all modules referenced by the middleware.
             * Note that they are guaranteed to be called in the order in which
             * the modules are declared.
             * @param pipe The pipeline's transitive state instance.
             */
            inline virtual void run(pipe_t& pipe) const override
            {
                utility::foreach(&module_t::run, m_sequence, pipe);
            }

            /**
             * Executes the finalization logic of all modules referenced by the
             * middleware. Note that, differently from the previous steps, this
             * method guarantees that modules are called in opposite order.
             * @param pipe The pipeline's transitive state instance.
             */
            inline virtual void after(pipe_t& pipe) const override
            {
                utility::rforeach(&module_t::after, m_sequence, pipe);
            }
    };

    /**
     * Configures a pipeline composed of modules and middlewares, and allows for
     * stateless, concurrent and thread-safe executions.
     * @tparam M The list of module types that are to execute.
     * @since 1.0
     */
    template <typename ...M>
    class runner_t
    {
        public:
            typedef pipeline::middleware_t<M...> executor_t;
            typedef typename executor_t::sequence_t sequence_t;

        protected:
            const executor_t m_executor = {};

        public:
            inline constexpr runner_t() = default;
            inline constexpr runner_t(const runner_t&) = default;
            inline constexpr runner_t(runner_t&&) = delete;

            /**
             * Configures a new pipeline runner with the modules it must execute.
             * @param modules The runner's modules instance list.
             */
            inline constexpr runner_t(M&&... modules)
              : m_executor (std::forward<decltype(modules)>(modules)...)
            {}

            inline runner_t& operator=(const runner_t&) = delete;
            inline runner_t& operator=(runner_t&&) = delete;

            /**
             * Runs the pipeline and returns the resulting objects.
             * @param pipe An optional pipeline's transitive state instance.
             * @return The resulting pipeline state.
             */
            inline auto run(const pipe_t& pipe = {}) const -> pipe_t
            {
                pipe_t state = pipe ? pipe : factory::memory::pointer::shared<state_t>();

                m_executor.before(state);
                m_executor.run(state);
                m_executor.after(state);

                return state;
            }
    };

    /*
     * Deduction guides for pipeline middlewares.
     * @since 1.0
     */
    template <typename ...M> middleware_t(M&&...) -> middleware_t<M...>;
    template <typename ...M> runner_t(M&&...) -> runner_t<M...>;
}

MUSEQA_END_NAMESPACE
