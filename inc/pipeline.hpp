/**
 * Multiple Sequence Alignment pipeline header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#include <utility>

#include <tuple.hpp>
#include <utils.hpp>
#include <pointer.hpp>
#include <commander.hpp>
#include <exception.hpp>

namespace msa
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
            static constexpr const char *module = "anonymous";  /// The corresponding module name.

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
            using previous = void;                  /// Indicates which module should execute before.
            using conduit = pipeline::conduit;      /// The module's conduit type.

            virtual auto check(const commander&) const -> bool = 0;
            virtual auto run(const commander&, const pointer<conduit>&) const -> pointer<conduit> = 0;
        };
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
                    && std::is_base_of<msa::pipeline::module, T>::value
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
        class runner final
        {
            static_assert(utils::all(std::is_base_of<module, T>()...), "pipeline can only handle modules");
            static_assert(utils::all(std::is_default_constructible<T>()...), "modules must default construct");
            static_assert(detail::pipeline::chainable<void, T...>(), "given modules cannot be chained");

            public:
                static constexpr size_t count = sizeof...(T);   /// The number of chained modules.

            private:
                using module_tuple = tuple<T...>;               /// The tuple of chained modules types.
                using conduit = pointer<pipeline::conduit>;     /// The return type expected from modules.

            public:
                /**
                 * Runs the pipeline and returns the last module's result.
                 * @param cmd The command line manager to pass to each module.
                 * @return The last module's resulting value.
                 */
                inline static auto run(const commander& cmd) -> conduit
                {
                    auto modules = module_tuple {};

                    auto lb1 = [&cmd](const bool prev, const module& mod) { return prev && mod.check(cmd); };
                    auto lb2 = [&cmd](const conduit& prev, const module& mod) { return mod.run(cmd, prev); };

                    if(!utils::foldl(lb1, true, modules))
                        throw exception {"command line arguments check failed"};
                    
                    return utils::foldl(lb2, conduit {}, modules);
                }
        };
    }
}
