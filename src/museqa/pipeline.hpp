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
     * A pipe is a connector between modules that are executed within a pipeline.
     * It is responsible for carrying stateful data down the pipeline chain.
     * @since 1.0
     */
    class pipe_t : private std::unordered_map<std::string, memory::pointer::shared_t<void>>
    {
        private:
            using entry_t = memory::pointer::shared_t<void>;
            using underlying_t = std::unordered_map<std::string, entry_t>;

        public:
            inline pipe_t() = default;
            inline pipe_t(const pipe_t&) = default;
            inline pipe_t(pipe_t&&) = default;

            inline pipe_t& operator=(const pipe_t&) = default;
            inline pipe_t& operator=(pipe_t&&) = default;

            /**
             * Retrieves data referenced by the given key from the pipe.
             * @tparam T The type of data to retrieve.
             * @param key The key to be requested from or created in the pipe.
             * @return The data retrieved from the pipe.
             */
            template <typename T = void>
            inline memory::pointer::shared_t<T>& retrieve(const std::string& key)
            {
                return reinterpret_cast<memory::pointer::shared_t<T>&>(
                    underlying_t::operator[](key)
                );
            }

            /**
             * Removes data referenced by a given key from the pipe. No errors or
             * exceptions will be raised if key is not found in the pipe.
             * @param key The key to be removed from the pipe.
             * @return Has the key been found and removed?
             */
            inline bool remove(const std::string& key)
            {
                return (bool) underlying_t::erase(key);
            }
    };

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
        virtual void before(memory::pointer::shared_t<pipe_t>& pipe) const { }

        /**
         * The method to execute the module's core logic. After the pipeline has
         * been successfully initialized, the module's core logic is can be run.
         * @param pipe The pipeline's transitive state instance.
         */
        virtual void run(memory::pointer::shared_t<pipe_t>& pipe) const = 0;

        /**
         * The method to run after the module's core logic. Within a pipeline, modules
         * have this method called in the opposite order to which they are defined.
         * @param pipe The pipeline's transitive state instance.
         */
        virtual void after(memory::pointer::shared_t<pipe_t>& pipe) const { }
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
    struct middleware_t : public pipeline::module_t
    {
        public:
            using sequence_t = utility::tuple_t<M...>;

        protected:
            sequence_t m_sequence = {};

        static_assert(
            utility::all(std::is_base_of<pipeline::module_t, M>::value...)
          , "pipeline can only run types that inherit from a module"
        );

        public:
            inline constexpr middleware_t() = default;
            inline constexpr middleware_t(const middleware_t&) = default;
            inline constexpr middleware_t(middleware_t&&) = default;

            /**
             * Configures a new pipeline middleware with the modules it must execute.
             * @param modules The middleware's modules instance list.
             */
            inline constexpr explicit middleware_t(M&&... modules)
              : m_sequence (std::forward<decltype(modules)>(modules)...)
            {}

            inline middleware_t& operator=(const middleware_t&) = default;
            inline middleware_t& operator=(middleware_t&&) = default;

            /**
             * Executes the validation or initialization logic of all modules referenced
             * by the middleware. Note that they are guaranteed to be called in
             * the order in which they are declared.
             * @param pipe The pipeline's transitive state instance.
             */
            inline virtual void before(memory::pointer::shared_t<pipe_t>& pipe) const override
            {
                utility::foreach(&module_t::before, m_sequence, pipe);
            }

            /**
             * Executes the core logic of all modules referenced by the middleware.
             * Note that they are guaranteed to be called in the order in which
             * the modules are declared.
             * @param pipe The pipeline's transitive state instance.
             */
            inline virtual void run(memory::pointer::shared_t<pipe_t>& pipe) const override
            {
                utility::foreach(&module_t::run, m_sequence, pipe);
            }

            /**
             * Executes the finalization logic of all modules referenced by the
             * middleware. Note that, differently from the previous steps, this
             * method guarantees that modules are called in opposite order.
             * @param pipe The pipeline's transitive state instance.
             */
            inline virtual void after(memory::pointer::shared_t<pipe_t>& pipe) const override
            {
                utility::rforeach(&module_t::after, m_sequence, pipe);
            }
    };

    /**
     * Manages the execution of a pipeline of modules and middlewares.
     * @tparam M The list of module types that are to execute.
     * @since 1.0
     */
    template <typename ...M>
    class runner_t final : private middleware_t<M...>
    {
        private:
            using underlying_t = middleware_t<M...>;

        public:
            using sequence_t = typename underlying_t::sequence_t;

        public:
            inline constexpr runner_t() = default;
            inline constexpr runner_t(const runner_t&) = default;
            inline constexpr runner_t(runner_t&&) = default;

            /**
             * Configures a new pipeline runner with the modules it must execute.
             * @param modules The runner's modules instance list.
             */
            inline constexpr explicit runner_t(M&&... modules)
              : underlying_t (std::forward<decltype(modules)>(modules)...)
            {}

            inline runner_t& operator=(const runner_t&) = default;
            inline runner_t& operator=(runner_t&&) = default;

            /**
             * Runs the pipeline and returns the resulting objects.
             * @param pipe An optional pipeline's transitive state instance.
             * @return The resulting pipeline state.
             */
            inline auto run(const memory::pointer::shared_t<pipe_t>& pipe = {}) const
            -> memory::pointer::shared_t<pipe_t>
            {
                memory::pointer::shared_t<pipe_t> result = pipe ? pipe :
                    factory::memory::pointer::shared<pipe_t>();

                underlying_t::before(result);
                underlying_t::run(result);
                underlying_t::after(result);

                return result;
            }
    };
}

MUSEQA_END_NAMESPACE
