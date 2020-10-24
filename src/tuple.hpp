/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A multiple type tuple and functional helpers.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>
#include <utility>

#include "utils.hpp"

namespace museqa
{
    namespace detail
    {
        namespace tuple
        {
            /**
             * Represents a tuple leaf, which holds one of a tuple's value.
             * @tparam I The index of the tuple's leaf.
             * @tparam T The type of the tuple's leaf member.
             * @since 0.1.1
             */
            template <size_t I, typename T>
            struct leaf
            {
                using element_type = T;         /// The leaf's element type.
                element_type value;             /// The value held by the current leaf.

                __host__ __device__ inline constexpr leaf() noexcept = default;
                __host__ __device__ inline constexpr leaf(const leaf&) noexcept = default;
                __host__ __device__ inline constexpr leaf(leaf&&) noexcept = default;

                /**
                 * Constructs a new tuple leaf.
                 * @param value The value to be held by the leaf.
                 */
                __host__ __device__ inline constexpr leaf(const element_type& value) noexcept
                :   value {value}
                {}

                /**
                 * Constructs a new tuple leaf.
                 * @param value The value to be moved into the leaf.
                 */
                __host__ __device__ inline constexpr leaf(element_type&& value) noexcept
                :   value {std::forward<decltype(value)>(value)}
                {}

                /**
                 * Constructs a new tuple leaf from foreign type.
                 * @tparam U A convertible type for leaf's value.
                 * @param value The value to be copied to the leaf.
                 */
                template <typename U>
                __host__ __device__ inline constexpr leaf(const U& value) noexcept
                :   value {static_cast<element_type>(value)}
                {}

                /**
                 * Constructs a new tuple leaf by copying a foreign leaf.
                 * @tparam U A convertible foreign type for leaf's value.
                 * @param other The leaf to copy contents from.
                 */
                template <size_t J, typename U>
                __host__ __device__ inline constexpr leaf(const leaf<J, U>& other) noexcept
                :   leaf {other.value}
                {}

                /**
                 * Constructs a new tuple leaf by moving a foreign leaf.
                 * @tparam U A convertible foreign type for leaf's value.
                 * @param other The leaf to move contents from.
                 */
                template <size_t J, typename U>
                __host__ __device__ inline constexpr leaf(leaf<J, U>&& other) noexcept
                :   leaf {std::forward<decltype(other.value)>(other.value)}
                {}

                __host__ __device__ inline leaf& operator=(const leaf&) = default;
                __host__ __device__ inline leaf& operator=(leaf&&) = default;
            };

            /**
             * Represents a tuple leaf, which holds a reference.
             * @tparam I The index of the tuple's leaf.
             * @tparam T The type of the tuple's leaf member.
             * @since 0.1.1
             */
            template <size_t I, typename T>
            struct leaf<I, T&>
            {
                using element_type = T&;        /// The leaf's element type.
                element_type value;             /// The value held by the current leaf.

                __host__ __device__ inline constexpr leaf() noexcept = default;
                __host__ __device__ inline constexpr leaf(const leaf&) noexcept = default;
                __host__ __device__ inline constexpr leaf(leaf&&) noexcept = default;

                /**
                 * Constructs a new tuple reference leaf.
                 * @param ref The reference to be held by the leaf.
                 */
                __host__ __device__ inline constexpr leaf(element_type ref) noexcept
                :   value {ref}
                {}

                /**
                 * Constructs a new tuple leaf from foreign type.
                 * @tparam U A convertible type for leaf's reference.
                 * @param ref The reference to be held by the leaf.
                 */
                template <
                        typename U
                    ,   typename = typename std::enable_if<std::is_convertible<U&, T&>::value>::type
                    >
                __host__ __device__ inline constexpr leaf(U& ref) noexcept
                :   value {ref}
                {}

                /**
                 * Constructs a new tuple leaf by copying a foreign leaf.
                 * @tparam U A convertible foreign type for leaf's reference.
                 * @param other The leaf to get reference from.
                 */
                template <size_t J, typename U>
                __host__ __device__ inline constexpr leaf(const leaf<J, U>& other) noexcept
                :   leaf {other.value}
                {}

                __host__ __device__ inline leaf& operator=(const leaf&) = default;
                __host__ __device__ inline leaf& operator=(leaf&&) = default;
            };

            /**#@+
             * The base for a type tuple.
             * @tparam I The indeces for the tuple members.
             * @tparam T The types of the tuple members.
             * @since 0.1.1
             */
            template <typename I, typename ...T>
            struct base;

            template <size_t ...I, typename ...T>
            struct base<indexer<I...>, T...> : public leaf<I, T>...
            {
                static constexpr size_t count = sizeof...(I);   /// The size of the tuple.

                __host__ __device__ inline constexpr base() noexcept = default;
                __host__ __device__ inline constexpr base(const base&) noexcept = default;
                __host__ __device__ inline constexpr base(base&&) noexcept = default;

                /**
                 * Forwards all values to the tuple's leaves.
                 * @tparam U The list of possibly foreign types for each base member.
                 * @param value The list of values for members.
                 */
                template <typename ...U>
                __host__ __device__ inline constexpr base(U&&... value) noexcept
                :   leaf<I, T> (std::forward<decltype(value)>(value))...
                {
                    static_assert(utils::all(std::is_convertible<U, T>()...), "tuple types not compatible");
                }

                __host__ __device__ inline base& operator=(const base&) = default;
                __host__ __device__ inline base& operator=(base&&) = default;
            };

            template <>
            struct base<indexer<>>
            {
                static constexpr size_t count = 0;      /// The size of the tuple.
            };
            /**#@-*/

            /**
             * Retrieves the requested tuple leaf and returns its value.
             * @param leaf The selected tuple leaf member.
             * @tparam I The requested leaf index.
             * @tparam T The type of requested leaf member.
             * @return The leaf's value.
             */
            template <size_t I, typename T>
            __host__ __device__ inline constexpr const T& get(const leaf<I, T>& leaf) noexcept
            {
                return leaf.value;
            }

            /**
             * Modifies the value held by a tuple leaf.
             * @tparam I The requested leaf index.
             * @tparam T The type of requested leaf member.
             * @param leaf The selected tuple leaf member.
             * @param value The value to copy to leaf.
             */
            template <size_t I, typename T>
            __host__ __device__ inline void set(leaf<I, T>& leaf, const T& value) noexcept
            {
                leaf.value = value;
            }

            /**
             * Modifies the value held by a tuple leaf.
             * @tparam I The requested leaf index.
             * @tparam T The type of requested leaf member.
             * @param leaf The selected tuple leaf member.
             * @param value The value to move to leaf.
             */
            template <size_t I, typename T>
            __host__ __device__ inline void set(leaf<I, T>& leaf, T&& value) noexcept
            {
                leaf.value = std::forward<decltype(value)>(value);
            }

            /**
             * Modifies the value held by a tuple leaf with a foreign type.
             * @tparam I The requested leaf index.
             * @tparam T The type of requested leaf member.
             * @tparam U The foreign type to copy to leaf.
             * @param leaf The selected tuple leaf member.
             * @param value The value to copy to leaf.
             */
            template <size_t I, typename T, typename U>
            __host__ __device__ inline void set(leaf<I, T>& leaf, const U& value) noexcept
            {
                leaf.value = value;
            }

            /**
             * Accesses the internal declared type of a tuple leaf.
             * @param (ignored) The leaf from which type will be extracted.
             */
            template <size_t I, typename T>
            __host__ __device__ inline constexpr auto type(const leaf<I, T>&) noexcept
            -> typename leaf<I, T>::element_type;
        }
    }

    /**
     * A tuple is responsible for holding a list of elements of possible different
     * types with a known number of elements.
     * @tparam T The tuple's list of member types.
     * @since 0.1.1
     */
    template <typename ...T>
    class tuple : public detail::tuple::base<indexer_g<sizeof...(T)>, T...>
    {
        protected:
            using indexer_type = indexer_g<sizeof...(T)>;   /// The tuple's indeces type.

        public:
            /**
             * Gets the type of a specific tuple element.
             * @tparam I The index of element to get type of.
             */
            template <size_t I>
            using element = decltype(detail::tuple::type<I>(std::declval<tuple>()));

        public:
            __host__ __device__ inline constexpr tuple() noexcept = default;
            __host__ __device__ inline constexpr tuple(const tuple&) noexcept = default;
            __host__ __device__ inline constexpr tuple(tuple&&) noexcept = default;

            using detail::tuple::base<indexer_type, T...>::base;

            /**
             * Creates a new tuple from a tuple of different base types.
             * @tparam U The types of foreign tuple instance to copy from.
             * @param other The tuple which values must be copied from.
             */
            template <typename ...U>
            __host__ __device__ inline tuple(const tuple<U...>& other) noexcept
            {
                operator=(other);
            }

            /**
             * Creates a new tuple from a tuple of different base types.
             * @tparam U The types of foreign tuple instance to copy from.
             * @param other The tuple the values must be copied from.
             */
            template <typename ...U>
            __host__ __device__ inline tuple(tuple<U...>&& other) noexcept
            {
                operator=(std::forward<decltype(other)>(other));
            }

            /**
             * Copies values from another tuple instance.
             * @param other The tuple the values must be copied from.
             * @return The current tuple instance.
             */
            __host__ __device__ inline tuple& operator=(const tuple& other)
            {
                return copy(indexer_type {}, other);
            }

            /**
             * Moves the values from another tuple instance.
             * @param other The tuple the values must be moved from.
             * @return The current tuple instance.
             */
            __host__ __device__ inline tuple& operator=(tuple&& other)
            {
                return copy(indexer_type {}, std::forward<decltype(other)>(other));
            }

            /**
             * Copies the values from a foreign tuple instance.
             * @tparam U The types of tuple instance to copy from.
             * @param other The tuple the values must be copied from.
             * @return The current tuple instance.
             */
            template <typename ...U>
            __host__ __device__ inline tuple& operator=(const tuple<U...>& other)
            {
                static_assert(sizeof...(U) == sizeof...(T), "tuples must have same cardinality");
                return copy(indexer_type {}, other);
            }

            /**
             * Copies the values from a foreign tuple instance.
             * @tparam U The types of tuple instance to copy from.
             * @param other The tuple the values must be copied from.
             * @return The current tuple instance.
             */
            template <typename ...U>
            __host__ __device__ inline tuple& operator=(tuple<U...>&& other)
            {
                static_assert(sizeof...(U) == sizeof...(T), "tuples must have same cardinality");
                return copy(indexer_type {}, std::forward<decltype(other)>(other));
            }

            /**
             * Gets value from member by index.
             * @tparam I The index of requested member.
             * @return The member's value.
             */
            template <size_t I>
            __host__ __device__ inline constexpr auto get() const noexcept -> const element<I>&
            {
                return detail::tuple::get<I>(*this);
            }

            /**
             * Sets a member value by its index.
             * @tparam I The index of requested member.
             * @tparam U The new value type.
             */
            template <size_t I, typename U>
            __host__ __device__ inline auto set(const U& value) noexcept -> void
            {
                detail::tuple::set<I>(*this, value);
            }

            /**
             * Sets a member value by its index.
             * @tparam I The index of requested member.
             * @tparam U The new value type.
             */
            template <size_t I, typename U>
            __host__ __device__ inline auto set(U&& value) noexcept
            -> typename std::enable_if<std::is_same<decltype(value), element<I>>::value>::type
            {
                detail::tuple::set<I>(*this, std::forward<decltype(value)>(value));
            }

        protected:
            /**
             * Recursion basis for copy operation.
             * @tparam U The foreign tuple type.
             * @return This object instance.
             */
            template <typename U>
            __host__ __device__ inline tuple& copy(indexer<>, const U&) noexcept
            {
                return *this;
            }

            /**
             * Copies values from a foreign tuple instance.
             * @tpatam I The first member index to be copied.
             * @tparam J The following member indeces to copy.
             * @tparam U The foreign tuple base types.
             * @return This object instance.
             */
            template <size_t I, size_t ...J, typename ...U>
            __host__ __device__ inline tuple& copy(indexer<I, J...>, const tuple<U...>& other) noexcept
            {
                set<I>(other.template get<I>());
                return copy(indexer<J...> {}, other);
            }

            /**
             * Moves values from a foreign tuple instance.
             * @tpatam I The first member index to be copied.
             * @tparam J The following member indeces to copy.
             * @tparam U The foreign tuple base types.
             * @return This object instance.
             */
            template <size_t I, size_t ...J, typename ...U>
            __host__ __device__ inline tuple& copy(indexer<I, J...>, tuple<U...>&& other) noexcept
            {
                set<I>(std::move(other.template get<I>()));
                return copy(indexer<J...> {}, std::forward<decltype(other)>(other));
            }
    };

    namespace detail
    {
        namespace tuple
        {
            /**
             * Creates a tuple with repeated types.
             * @tparam T The type to repeat.
             * @tparam I The number of times to repeat the type.
             */
            template <typename T, size_t ...I>
            constexpr auto repeater(indexer<I...>) noexcept
            -> museqa::tuple<identity<T, I>...>;
        }
    }

    /**
     * Creates a tuple with repeated types.
     * @tparam T The type to be repeated.
     * @tparam N The number of times the type shall repeat.
     * @since 0.1.1
     */
    template <typename T, size_t N>
    class ntuple : public decltype(detail::tuple::repeater<T>(indexer_g<N>()))
    {
        protected:
            using indexer_type = indexer_g<N>;      /// The list of index for current tuple.

        public:
            using element_type = pure<T>;           /// The tuple's element type.
            using underlying_tuple = decltype(detail::tuple::repeater<T>(indexer_type {}));

        public:
            __host__ __device__ inline constexpr ntuple() noexcept = default;
            __host__ __device__ inline constexpr ntuple(const ntuple&) noexcept = default;
            __host__ __device__ inline constexpr ntuple(ntuple&&) noexcept = default;

            using underlying_tuple::tuple;

            /**
             * Initializes a new tuple from an array.
             * @param arr The array to initialize tuple.
             */
            __host__ __device__ inline constexpr ntuple(element_type *arr) noexcept
            :   underlying_tuple {extract(indexer_type {}, arr)}
            {}

            /**
             * Initializes a new tuple from a const array.
             * @param arr The array to initialize tuple.
             */
            template <
                    typename U = T
                ,   typename = typename std::enable_if<!std::is_reference<U>::value>::type
                >
            __host__ __device__ inline constexpr ntuple(const element_type *arr) noexcept
            :   underlying_tuple {extract(indexer_type {}, arr)}
            {}

            __host__ __device__ inline ntuple& operator=(const ntuple&) = default;
            __host__ __device__ inline ntuple& operator=(ntuple&&) = default;

            using underlying_tuple::operator=;

        protected:
            /**
             * A helper function to map array values to the underlying tuple.
             * @param arr The array to inline.
             * @return The created tuple.
             */
            template <size_t ...I, typename U>
            __host__ __device__ inline static constexpr auto extract(indexer<I...>, U *arr) noexcept
            -> underlying_tuple
            {
                return {arr[I]...};
            }
    };

    /**
     * The type of a tuple's element.
     * @tparam T The target tuple.
     * @tparam I The index of tuple element.
     * @since 0.1.1
     */
    template <typename T, size_t I>
    using tuple_element = typename T::template element<I>;

    namespace detail
    {
        namespace tuple
        {
            /**#@+
             * Gathers variable or array references into a tuple, allowing them
             * to capture values directly from value tuples.
             * @tparam T The gathered variables types.
             * @tparam N When an array, the size must be fixed.
             * @param arg The gathered variables references.
             * @return The new tuple of references.
             */
            template <typename ...T>
            __host__ __device__ inline constexpr auto tie(T&... arg) noexcept -> museqa::tuple<T&...>
            {
                return {arg...};
            }

            template <typename T, size_t N>
            __host__ __device__ inline constexpr auto tie(T (&arg)[N]) noexcept -> museqa::ntuple<T&, N>
            {
                return {arg};
            }
            /**#@-*/

            /**
             * Retrieves and returns the value of the first leaf of a tuple.
             * @tparam I The list of tuple indeces to move to the new tuple.
             * @tparam T The type of tuple's element members.
             * @param tp The tuple to access its first element.
             * @return The head value of tuple.
             */
            template <size_t ...I, typename ...T>
            __host__ __device__ inline constexpr auto head(const base<indexer<0, I...>, T...>& tp) noexcept
            -> decltype(get<0>(std::declval<decltype(tp)>()))
            {
                return get<0>(tp);
            }

            /**
             * Retrieves and returns the value of the last leaf of a tuple.
             * @tparam I The list of tuple indeces to move to the new tuple.
             * @tparam T The type of tuple's element members.
             * @param tp The tuple to access its last element.
             * @return The last value of tuple.
             */
            template <size_t ...I, typename ...T>
            __host__ __device__ inline constexpr auto last(const base<indexer<0, I...>, T...>& tp) noexcept
            -> decltype(get<sizeof...(I)>(std::declval<decltype(tp)>()))
            {
                return get<sizeof...(I)>(tp);
            }

            /**
             * Removes the last element of the tuple and returns the rest.
             * @tparam I The list of tuple indeces to move to the new tuple.
             * @tparam T The list of types to move to the new tuple.
             * @param tp The tuple to have its last element removed.
             * @return The init of given tuple.
             */
            template <size_t ...I, typename ...T>
            __host__ __device__ inline constexpr auto init(const base<indexer<0, I...>, T...>& tp) noexcept
            -> museqa::tuple<decltype(type<I - 1>(std::declval<decltype(tp)>()))...>
            {
                return {get<I - 1>(tp)...};
            }

            /**
             * Removes the first element of the tuple and returns the rest.
             * @tparam I The list of tuple indeces to move to the new tuple.
             * @tparam T The list of types to move to the new tuple.
             * @param tp The tuple to have its head removed.
             * @return The tail of given tuple.
             */
            template <size_t ...I, typename ...T>
            __host__ __device__ inline constexpr auto tail(const base<indexer<0, I...>, T...>& tp) noexcept
            -> museqa::tuple<decltype(type<I>(std::declval<decltype(tp)>()))...>
            {
                return {get<I>(tp)...};
            }

            /**#@+
             * Concatenates a list of tuples into a single one.
             * @param a The first tuple to concatenate.
             * @param b The second tuple to concatenate.
             * @param zero The base tuple, which no other need to concatenate with.
             * @param tail The following tuples to concatenate.
             * @return A concatenated tuple of all others.
             */
            template <typename ...T>
            __host__ __device__ inline constexpr auto concat(const museqa::tuple<T...>& zero) noexcept
            -> museqa::tuple<T...>
            {
                return zero;
            }

            template <size_t ...I, size_t ...J, typename ...T, typename ...U, typename ...R>
            __host__ __device__ inline constexpr auto concat(
                    const base<indexer<I...>, T...>& a
                ,   const base<indexer<J...>, U...>& b
                ,   const R&... tail
                ) noexcept
            -> decltype(concat(std::declval<museqa::tuple<T..., U...>>(), std::declval<R>()...))
            {
                using merged_tuple = museqa::tuple<T..., U...>;
                return concat(merged_tuple {get<I>(a)..., get<J>(b)...}, tail...);
            }
            /**#@-*/

            /**
             * Applies an operator to all tuple's elements.
             * @tparam F The functor type to apply.
             * @tparam A The types of extra arguments.
             * @tparam I The tuple's indeces.
             * @tparam T The tuple's types.
             * @param lambda The functor to apply to tuple.
             * @param zero The base first operator value.
             * @param tp The tuple to apply functor to.
             * @return The new tuple.
             */
            template <typename F, typename ...A, size_t ...I, typename ...T>
            __host__ __device__ inline constexpr auto apply(
                    F&& lambda
                ,   const base<indexer<I...>, T...>& tp
                ,   const A&... args
                )
            -> museqa::tuple<decltype(lambda(std::declval<T>(), std::declval<A>()...))...>
            {
                return {lambda(get<I>(tp), args...)...};
            }

            /**#@+
             * Performs a left fold, or reduction in the given tuple.
             * @tparam F The functor type to combine the values.
             * @tparam B The base and return fold value type.
             * @tparam T The tuple value types.
             * @param lambda The functor used to created the new elements.
             * @param zero The base folding value.
             * @param tp The tuple to fold.
             * @return The final value.
             */
            template <typename F, typename B>
            __host__ __device__ inline constexpr B foldl(F&&, const B& zero, const museqa::tuple<>&)
            {
                return zero;
            }

            template <typename F, typename B, typename ...T>
            __host__ __device__ inline constexpr B foldl(F&& lambda, const B& zero, const museqa::tuple<T...>& tp)
            {
                return foldl(lambda, lambda(zero, head(tp)), tail(tp));
            }
            /**#@-*/

            /**#@+
             * Performs a right fold, or reduction in the given tuple.
             * @tparam F The functor type to combine the values.
             * @tparam B The base and return fold value type.
             * @tparam T The tuple value types.
             * @param lambda The functor used to created the new elements.
             * @param zero The base folding value.
             * @param tp The tuple to fold.
             * @return The final value.
             */
            template <typename F, typename B>
            __host__ __device__ inline constexpr B foldr(F&&, const B& zero, const museqa::tuple<>&)
            {
                return zero;
            }

            template <typename F, typename B, typename ...T>
            __host__ __device__ inline constexpr B foldr(F&& lambda, const B& zero, const museqa::tuple<T...>& tp)
            {
                return lambda(head(tp), foldr(lambda, zero, tail(tp)));
            }
            /**#@-*/

            /**#@+
             * Applies a left fold and returns all intermediate and final steps.
             * @tparam F The functor type to combine the values.
             * @tparam B The base and return fold value type.
             * @tparam I The tuple index counter.
             * @tparam T The tuple types.
             * @param lambda The functor used to created the new elements.
             * @param zero The base folding value.
             * @param tp The tuple to fold.
             * @return The intermediate and final values.
             */
            template <typename F, typename B>
            __host__ __device__ inline constexpr auto scanl(F&&, const B& zero, const base<indexer<>>&)
            -> museqa::tuple<B>
            {
                return {zero};
            }

            template <typename F, typename B, typename ...T, size_t ...I>
            __host__ __device__ inline constexpr auto scanl(
                    F&& lambda
                ,   const B& zero
                ,   const base<indexer<I...>, T...>& tp
                )
            -> museqa::tuple<B, identity<B, I>...>
            {
                return {zero, get<I>(scanl(lambda, lambda(zero, head(tp)), tail(tp)))...};
            }
            /**#@-*/

            /**#@+
             * Applies a right fold and returns all intermediate and final steps.
             * @tparam F The functor type to combine the values.
             * @tparam B The base and return fold value type.
             * @tparam I The tuple index counter.
             * @tparam T The tuple types.
             * @param lambda The functor used to created the new elements.
             * @param zero The base folding value.
             * @param t The tuple to fold.
             * @return The intermediate and final values.
             */
            template <typename F, typename B>
            __host__ __device__ inline constexpr auto scanr(F&&, const B& zero, const base<indexer<>>&)
            -> museqa::tuple<B>
            {
                return {zero};
            }

            template <typename F, typename B, typename ...T, size_t ...I>
            __host__ __device__ inline constexpr auto scanr(
                    F&& lambda
                ,   const B& zero
                ,   const base<indexer<I...>, T...>& tp
                )
            -> museqa::tuple<identity<B, I>..., B>
            {
                return {get<I>(scanr(lambda, lambda(head(tp), zero), tail(tp)))..., zero};
            }
            /**#@-*/

            /**
             * Zips two tuples together, creating a new second order tuple with internal
             * tuples of intercalated types of the original tuple.
             * @tparam I The list of tuple indeces in the input tuples.
             * @tparam T The list of types in the first input tuple.
             * @tparam U The list of types in the second input tuple.
             * @param a The first input tuple.
             * @param b The second input tuple.
             * @param The new second order tuple with intercalated elements.
             */
            template <size_t ...I, typename ...T, typename ...U>
            __host__ __device__ inline constexpr auto zip(
                    const base<indexer<I...>, T...>& a
                ,   const base<indexer<I...>, U...>& b
                )
            -> museqa::tuple<museqa::tuple<T, U>...>
            {
                return {museqa::tuple<T, U> {get<I>(a), get<I>(b)}...};
            }

            /**
             * Creates a new tuple with elements calculated from the given functor and
             * the elements of input tuples occuring at the same position in both tuples.
             * @tparam F The functor type to use to construct the new tuple.
             * @tparam I The list of tuple indeces in the input tuples.
             * @tparam T The list of types in the first input tuple.
             * @tparam U The list of types in the second input tuple.
             * @param lambda The functor used to created the new elements.
             * @param a The first input tuple.
             * @param b The second input tuple.
             * @param The new tuple with the calculated elements.
             */
            template <typename F, size_t ...I, typename ...T, typename ...U>
            __host__ __device__ inline constexpr auto zipwith(
                    F&& lambda
                ,   const base<indexer<I...>, T...>& a
                ,   const base<indexer<I...>, U...>& b
                )
            -> museqa::tuple<decltype(lambda(std::declval<T>(), std::declval<U>()))...>
            {
                return {lambda(get<I>(a), get<I>(b))...};
            }
        }
    }

    namespace utils
    {
        /*
         * Including tuple functions in the utility namespace. This gives these
         * functions a higher accessibility, usability and shorter names.
         */
        using detail::tuple::tie;
        using detail::tuple::zip;
        using detail::tuple::head;
        using detail::tuple::last;
        using detail::tuple::init;
        using detail::tuple::tail;
        using detail::tuple::apply;
        using detail::tuple::foldl;
        using detail::tuple::foldr;
        using detail::tuple::scanl;
        using detail::tuple::scanr;
        using detail::tuple::concat;
        using detail::tuple::zipwith;
    }
}