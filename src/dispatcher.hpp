/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements a string and index key-value dispatcher.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#include <map>
#include <string>
#include <vector>

#include "exception.hpp"

namespace museqa
{
    /**
     * Maps elements of given type to a string identificator.
     * @tparam T The type of element to dispatch.
     * @since 0.1.1
     */
    template <typename T>
    class dispatcher : protected std::map<std::string, T>
    {
        public:
            using element_type = T;                         /// The type of element to dispatch.

        protected:
            using underlying_type = std::map<std::string, element_type>;
            using pair_type = typename underlying_type::value_type;

        protected:
            std::vector<std::string> m_keylist;             /// The list of keys known to dispatcher.

        public:
            inline dispatcher() noexcept = default;
            inline dispatcher(const dispatcher&) = default;
            inline dispatcher(dispatcher&&) = default;

            /**
             * Initializes a dispatcher with a map of dispatchable entries.
             * @param entries The dispatcher's entries.
             */
            inline dispatcher(const underlying_type& entries)
            :   underlying_type {entries}
            {
                for(const auto& entry : entries)
                    m_keylist.push_back(entry.first);
            }

            /**
             * Allows the dispatcher to be initialized with some syntactic sugar.
             * @param entries The dispatcher's entries.
             */
            inline dispatcher(const std::initializer_list<pair_type>& entries)
            :   dispatcher {underlying_type {entries}}
            {}

            inline dispatcher& operator=(const dispatcher&) = default;
            inline dispatcher& operator=(dispatcher&&) = default;

            /**
             * Dispatches an element addressed by given key.
             * @param key The key to find element to dispatch.
             * @return The dispatched element.
             */
            inline const element_type& operator[](const std::string& key) const
            {
                const auto& selected = this->find(key);
                enforce(selected != this->end(), "dispatcher could not find key '%s'", key);
                return selected->second;
            }

            /**
             * Checks whether a given key is known by the dispatcher.
             * @param key The key to check existence of in dispatcher.
             * @return Is the given key known?
             */
            inline bool has(const std::string& key) const noexcept
            {
                return this->find(key) != this->end();
            }

            /**
             * Informs the list of registered key entries in dispatcher.
             * @return The list of registered keys.
             */
            inline const std::vector<std::string>& list() const noexcept
            {
                return m_keylist;
            }
    };
}
