/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A general tuple type abstraction implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>
#include <utility>

#include <museqa/utility.hpp>
#include <museqa/utility/indexer.hpp>

namespace museqa
{
    namespace utility
    {
        /**
         * A tuple represents an indexable sequential list of elements of possibly
         * different types. In comparision with a plain struct containing elements
         * of similar types, the tuple must require the same amount of memory and
         * its elements cannot be accessed by field names but offset.
         * @tparam T The tuple's sequence of element types.
         * @since 1.0
         */
        template <typename ...T>
        class tuple : public tuple<typename indexer<sizeof...(T)>::type, T...>
        {
          private:
            typedef typename indexer<sizeof...(T)>::type indexer_type;  /// The tuple's types index sequence.
            typedef tuple<indexer_type, T...> underlying_type;          /// The tuple's underlying implementation.

          public:
            __host__ __device__ inline constexpr tuple() noexcept = default;
            __host__ __device__ inline constexpr tuple(const tuple&) noexcept = default;
            __host__ __device__ inline constexpr tuple(tuple&&) noexcept = default;

            using underlying_type::tuple;

            /**
             * Creates a new tuple instance from a tuple of foreign types.
             * @tparam U The types of foreign tuple instance to copy from.
             * @param other The tuple which values must be copied from.
             */
            template <typename ...U>
            __host__ __device__ inline constexpr tuple(const tuple<U...>& other)
              : underlying_type {other}
            {}

            /**
             * Creates a new tuple instance by moving from a tuple of foreign types.
             * @tparam U The types of foreign tuple instance to move from.
             * @param other The tuple the values must be moved from.
             */
            template <typename ...U>
            __host__ __device__ inline constexpr tuple(tuple<U...>&& other)
              : underlying_type {std::forward<decltype(other)>(other)}
            {}

            __host__ __device__ inline tuple& operator=(const tuple&) = default;
            __host__ __device__ inline tuple& operator=(tuple&&) = default;

            using underlying_type::operator=;
        };

        namespace impl
        {
            /**
             * Represents a tuple leaf, which holds one of a tuple's value.
             * @tparam I The tuple's leaf's index offset.
             * @tparam T The tuple's leaf's content type.
             * @since 1.0
             */
            template <size_t I, typename T>
            struct leaf
            {
                typedef T element_type;         /// The leaf's content type.
                element_type value {};          /// The value contained by the current leaf.

                __host__ __device__ inline constexpr leaf() noexcept = default;
                __host__ __device__ inline constexpr leaf(const leaf&) noexcept = default;
                __host__ __device__ inline constexpr leaf(leaf&&) noexcept = default;

                /**
                 * Constructs a new tuple leaf.
                 * @param value The value to be contained by the leaf.
                 */
                __host__ __device__ inline constexpr leaf(const element_type& value) noexcept
                  : value {value}
                {}

                /**
                 * Constructs a new tuple leaf by moving a value.
                 * @param value The value to be moved into the leaf.
                 */
                __host__ __device__ inline constexpr leaf(element_type&& value) noexcept
                  : value {std::forward<decltype(value)>(value)}
                {}

                /**
                 * Constructs a new tuple leaf from a value of foreign type.
                 * @tparam U A convertible type for the leaf's value.
                 * @param value The value to be copied into the leaf.
                 */
                template <
                    typename U
                  , typename = typename std::enable_if<std::is_convertible<U, T>::value>::type
                >
                __host__ __device__ inline constexpr leaf(const U& value) noexcept
                  : value {static_cast<element_type>(value)}
                {}

                /**
                 * Constructs a new tuple leaf by copying a foreign leaf.
                 * @tparam U A convertible foreign type for the leaf's value.
                 * @param other The leaf to copy contents from.
                 */
                template <size_t J, typename U>
                __host__ __device__ inline constexpr leaf(const leaf<J, U>& other) noexcept
                  : leaf {other.value}
                {}

                /**
                 * Constructs a new tuple leaf by moving a foreign leaf.
                 * @tparam U A convertible foreign type for the leaf's value.
                 * @param other The leaf to move the contents from.
                 */
                template <size_t J, typename U>
                __host__ __device__ inline constexpr leaf(leaf<J, U>&& other) noexcept
                  : leaf {std::forward<decltype(other.value)>(other.value)}
                {}

                __host__ __device__ inline leaf& operator=(const leaf&) = default;
                __host__ __device__ inline leaf& operator=(leaf&&) = default;
            };

            /**
             * Represents a tuple leaf that holds a reference to a value.
             * @tparam I The tuple's leaf's index offset.
             * @tparam T The tuple's leaf's reference type.
             * @since 1.0
             */
            template <size_t I, typename T>
            struct leaf<I, T&>
            {
                typedef T& element_type;        /// The leaf's reference type.
                element_type value;             /// The reference contained by the current leaf.

                __host__ __device__ inline constexpr leaf() noexcept = delete;
                __host__ __device__ inline constexpr leaf(const leaf&) noexcept = default;
                __host__ __device__ inline constexpr leaf(leaf&&) noexcept = delete;

                /**
                 * Constructs a new tuple reference leaf.
                 * @param ref The reference to be held by the leaf.
                 */
                __host__ __device__ inline constexpr leaf(element_type ref) noexcept
                  : value {ref}
                {}

                /**
                 * Constructs a new tuple leaf from foreign reference type.
                 * @tparam U A convertible type for the leaf's reference.
                 * @param ref The reference to be held by the leaf.
                 */
                template <
                    typename U
                  , typename = typename std::enable_if<std::is_convertible<U&, T&>::value>::type
                >
                __host__ __device__ inline constexpr leaf(U& ref) noexcept
                  : value {ref}
                {}

                /**
                 * Constructs a new tuple leaf by copying a foreign reference leaf.
                 * @tparam U A convertible foreign type for the leaf's reference.
                 * @param other The leaf to get the reference from.
                 */
                template <size_t J, typename U>
                __host__ __device__ inline constexpr leaf(const leaf<J, U>& other) noexcept
                  : leaf {other.value}
                {}

                __host__ __device__ inline leaf& operator=(const leaf&) = delete;
                __host__ __device__ inline leaf& operator=(leaf&&) = delete;
            };

            /**
             * Retrieves the requested tuple leaf and returns its value.
             * @tparam I The requested leaf index.
             * @tparam T The type of the requested leaf member.
             * @param leaf The selected tuple leaf member.
             * @return The leaf's value.
             */
            template <size_t I, typename T>
            __host__ __device__ inline constexpr const T& get(const leaf<I, T>& leaf) noexcept
            {
                return leaf.value;
            }

            /**
             * Retrieves the requested tuple leaf and moves its value.
             * @tparam I The requested leaf index.
             * @tparam T The type of the requested leaf member.
             * @param leaf The selected tuple leaf member to be moved.
             * @return The moving leaf's value.
             */
            template <size_t I, typename T>
            __host__ __device__ inline constexpr T&& get(leaf<I, T>&& leaf) noexcept
            {
                return std::move(leaf.value);
            }

            /**
             * Modifies a tuple leaf by moving a value into it.
             * @tparam I The requested leaf index.
             * @tparam T The type of requested leaf member.
             * @param leaf The selected tuple leaf member.
             * @param value The value to move into the leaf.
             */
            template <size_t I, typename T, typename U>
            __host__ __device__ inline const T& set(leaf<I, T>& leaf, U&& value)
            {
                return leaf.value = std::forward<decltype(value)>(value);
            }

            /**
             * Creates a tuple with repeated types.
             * @tparam T The type to be repeated as tuple elements.
             * @tparam I The tuple's type index sequence.
             */
            template <typename T, size_t ...I>
            __host__ __device__ constexpr auto repeater(indexer<I...>) noexcept
            -> tuple<identity<T, I>...>;

            /**
             * Accesses the internal declared type of a tuple leaf.
             * @tparam I The index of the leaf to be accessed in the tuple.
             * @tparam T The extracted tuple element type.
             */
            template <size_t I, typename T>
            __host__ __device__ constexpr auto type(leaf<I, T>) noexcept
            -> typename leaf<I, T>::element_type;
        }

        /**
         * The base type for a tuple.
         * @tparam I The sequence index type for the tuple elements' types.
         * @tparam T The list of the tuple's elements' types.
         * @since 1.0
         */
        template <size_t ...I, typename ...T>
        class tuple<indexer<I...>, T...> : public impl::leaf<I, T>...
        {
          private:
            typedef indexer<I...> indexer_type;             /// The tuple's types index sequence.

          public:
            /**
             * Retrieves the type of a specific tuple element by its index.
             * @tparam I The requested element index.
             * @since 1.0
             */
            template <size_t J>
            using element = decltype(impl::type<J>(std::declval<tuple>()));

          public:
            static constexpr size_t count = sizeof...(I);   /// The total number of elements in the tuple.

          public:
            __host__ __device__ inline constexpr tuple() noexcept = default;
            __host__ __device__ inline constexpr tuple(const tuple&) noexcept = default;
            __host__ __device__ inline constexpr tuple(tuple&&) noexcept = default;

            /**
             * Creates a new tuple instance from a list of values.
             * @tparam U The values' types to build the tuple from.
             * @param value The list of concrete values to create the tuple with.
             */
            template <
                typename ...U
              , typename = typename std::enable_if<sizeof...(T) == sizeof...(U)>::type
            >
            __host__ __device__ inline constexpr tuple(U&&... value)
              : impl::leaf<I, T> (value)...
            {}

            /**
             * Creates a new tuple instance from a tuple of foreign types.
             * @tparam U The types of foreign tuple instance to copy from.
             * @param other The tuple which values must be copied from.
             */
            template <typename ...U>
            __host__ __device__ inline constexpr tuple(const tuple<indexer_type, U...>& other)
              : impl::leaf<I, T> (static_cast<impl::leaf<I, U>>(other))...
            {}

            /**
             * Creates a new tuple instance by moving a tuple of foreign types.
             * @tparam U The types of foreign tuple instance to move from.
             * @param other The tuple which values must be moved from.
             */
            template <typename ...U>
            __host__ __device__ inline constexpr tuple(tuple<indexer_type, U...>&& other)
              : impl::leaf<I, T> (std::forward<impl::leaf<I, U>>(other))...
            {}

            /**
             * Copies the values from a different tuple instance.
             * @param other The tuple the values must be copied from.
             * @return The current tuple instance.
             */
            __host__ __device__ inline tuple& operator=(const tuple& other)
            {
                return swallow(*this, impl::set<I>(*this, impl::get<I>(other))...);
            }

            /**
             * Moves the values from a different tuple instance.
             * @param other The tuple the values must be moved from.
             * @return The current tuple instance.
             */
            __host__ __device__ inline tuple& operator=(tuple&& other)
            {
                return swallow(*this, impl::set<I>(*this, impl::get<I>(std::forward<decltype(other)>(other)))...);
            }

            /**
             * Copies the values from a foreign tuple instance.
             * @tparam U The types of foreign tuple instance to copy from.
             * @param other The tuple the values must be copied from.
             * @return The current tuple instance.
             */
            template <typename ...U>
            __host__ __device__ inline tuple& operator=(const tuple<indexer_type, U...>& other)
            {
                return swallow(*this, impl::set<I>(*this, impl::get<I>(other))...);
            }

            /**
             * Moves the values from a foreign tuple instance.
             * @tparam U The types of foreign tuple instance to move from.
             * @param other The tuple the values must be moved from.
             * @return The current tuple instance.
             */
            template <typename ...U>
            __host__ __device__ inline tuple& operator=(tuple<indexer_type, U...>&& other)
            {
                return swallow(*this, impl::set<I>(*this, impl::get<I>(std::forward<decltype(other)>(other)))...);
            }

            /**
             * Retrieves the value of a tuple member by its index.
             * @tparam J The requested member's index.
             * @return The member's value.
             */
            template <size_t J>
            __host__ __device__ inline constexpr auto get() const noexcept -> const element<J>&
            {
                return impl::get<J>(*this);
            }

            /**
             * Updates the value of a tuple member by its index.
             * @tparam J The requested member's index.
             * @tparam U The member's new value's type.
             */
            template <size_t J, typename U>
            __host__ __device__ inline void set(U&& value)
            {
                impl::set<J>(*this, std::forward<decltype(value)>(value));
            }
        };

        /**
         * A tuple containing all elements of a single type can have its representation
         * simplified by a N-tuple, which works in similar ways to an array, but
         * with compile-time size delimitation and validations.
         * @tparam T The tuple's elements' type.
         * @tparam N The number of elements in the tuple.
         * @since 1.0
         */
        template <typename T, size_t N>
        class ntuple : public decltype(impl::repeater<T>(typename indexer<N>::type ()))
        {
          private:
            typedef typename indexer<N>::type indexer_type;
            typedef decltype(impl::repeater<T>(indexer_type {})) underlying_type;

          public:
            using element_type = T;         /// The tuple's elements' type.

          public:
            __host__ __device__ inline constexpr ntuple() noexcept = default;
            __host__ __device__ inline constexpr ntuple(const ntuple&) noexcept = default;
            __host__ __device__ inline constexpr ntuple(ntuple&&) noexcept = default;

            using underlying_type::tuple;

            /**
             * Creates a new tuple from a raw array.
             * @tparam U The foreign array type to create tuple from.
             * @param array The array to initialize the tuple's values from.
             */
            template <typename U>
            __host__ __device__ inline constexpr ntuple(U array[]) noexcept
              : ntuple {indexer_type {}, array}
            {}

            __host__ __device__ inline ntuple& operator=(const ntuple&) = default;
            __host__ __device__ inline ntuple& operator=(ntuple&&) = default;

            using underlying_type::operator=;

          private:
            /**
             * Creates a new tuple by inlining an array.
             * @tparam U The foreign array type to create tuple from.
             * @tparam I The tuple's sequence index for inlining the array.
             * @param array The array to inline.
             */
            template <typename U, size_t ...I>
            __host__ __device__ inline constexpr ntuple(indexer<I...>, U array[])
              : underlying_type {array[I]...}
            {}
        };

        /**
         * The tuple composed of exactly two elements is a pair. In a pair, each
         * of the elements can be more easily accessed by aliased methods.
         * @tparam T The first element's type.
         * @tparam U The second element's type.
         * @since 1.0
         */
        template <typename T, typename U>
        class pair : public tuple<T, U>
        {
          private:
            typedef tuple<T, U> underlying_type;

          public:
            __host__ __device__ inline constexpr pair() noexcept = default;
            __host__ __device__ inline constexpr pair(const pair&) noexcept = default;
            __host__ __device__ inline constexpr pair(pair&&) noexcept = default;

            using underlying_type::tuple;

            __host__ __device__ inline pair& operator=(const pair&) = default;
            __host__ __device__ inline pair& operator=(pair&&) = default;

            using underlying_type::operator=;

            /**
             * Retrieves the first element of the pair.
             * @return The pair's first element's reference.
             */
            __host__ __device__ inline constexpr auto first() const noexcept -> const T&
            {
                return impl::get<0>(*this);
            }

            /**
             * Retrieves the second element of the pair.
             * @return The pair's second element's reference.
             */
            __host__ __device__ inline constexpr auto second() const noexcept -> const U&
            {
                return impl::get<1>(*this);
            }
        };

        /**
         * The type of an element in a tuple.
         * @tparam T The target tuple type.
         * @tparam I The index of tuple element.
         * @since 1.0
         */
        template <typename T, size_t I>
        using tuple_element = typename T::template element<I>;

        /**#@+
         * Gathers variable or array references into a tuple instance, allowing
         * them to capture values directly from value tuples.
         * @tparam T The gathered variables types.
         * @tparam N When an array, the size must be fixed.
         * @param ref The gathered variables references.
         * @return The new tuple of references.
         */
        template <typename ...T>
        __host__ __device__ inline constexpr auto tie(T&... ref) noexcept -> tuple<T&...>
        {
            return {ref...};
        }

        template <typename T, size_t N>
        __host__ __device__ inline constexpr auto tie(T (&ref)[N]) noexcept -> ntuple<T&, N>
        {
            return {ref};
        }
        /**#@-*/

        /**
         * Retrieves and returns the value of the first leaf of a tuple.
         * @tparam I The tuple sequence indeces to match from tuple.
         * @tparam T The list of tuple's element members types.
         * @param t The tuple to get the first element from.
         * @return The head value of tuple.
         */
        template <size_t ...I, typename ...T>
        __host__ __device__ inline constexpr auto head(const tuple<indexer<I...>, T...>& t) noexcept
        -> decltype(impl::get<0>(t))
        {
            return impl::get<0>(t);
        }

        /**
         * Retrieves and returns the value of the last leaf of a tuple.
         * @tparam I The tuple sequence indeces to match from tuple.
         * @tparam T The list of tuple's element members types.
         * @param t The tuple to get the last element from.
         * @return The last value of tuple.
         */
        template <size_t ...I, typename ...T>
        __host__ __device__ inline constexpr auto last(const tuple<indexer<I...>, T...>& t) noexcept
        -> decltype(impl::get<sizeof...(T) - 1>(t))
        {
            return impl::get<sizeof...(T) - 1>(t);
        }

        /**
         * Returns a tuple with its last leaf removed.
         * @tparam I The tuple sequence indeces to match from tuple.
         * @tparam T The list of tuple's element members types.
         * @param t The tuple to have its last element removed.
         * @return The new tuple with removed end.
         */
        template <size_t ...I, typename ...T>
        __host__ __device__ inline constexpr auto init(const tuple<indexer<0, I...>, T...>& t) noexcept
        -> tuple<decltype(impl::type<I - 1>(t))...>
        {
            return {impl::get<I - 1>(t)...};
        }

        /**
         * Returns a tuple with its first leaf removed.
         * @tparam I The tuple sequence indeces to match from tuple.
         * @tparam T The list of tuple's element members types.
         * @param t The tuple to have its first element removed.
         * @return The new tuple with removed head.
         */
        template <size_t ...I, typename ...T>
        __host__ __device__ inline constexpr auto tail(const tuple<indexer<0, I...>, T...>& t) noexcept
        -> tuple<decltype(impl::type<I>(t))...>
        {
            return {impl::get<I>(t)...};
        }

        /**
         * Concatenates multiple tuples together into a single one.
         * @tparam I The first tuple sequence indeces.
         * @tparam J The second tuple sequence indeces.
         * @tparam T The first tuple's element members types.
         * @tparam U The second tuple's element members types.
         * @tparam R The following tuple types to concatenate.
         * @param a The first tuple to concatenate.
         * @param b The second tuple to concatenate.
         * @param tail The following tuples to concatenate.
         * @return The resulting concatenated tuple.
         */
        template <size_t ...I, size_t ...J, typename ...T, typename ...U, typename ...R>
        __host__ __device__ inline constexpr auto concat(
            const tuple<indexer<I...>, T...>& a
          , const tuple<indexer<J...>, U...>& b
          , const R&... tail
        ) noexcept -> decltype(concat(std::declval<tuple<T..., U...>>(), tail...))
        {
            return concat(tuple<T..., U...> {impl::get<I>(a)..., impl::get<J>(b)...}, tail...);
        }

        /**
         * The recursion base for a tuple concatenation.
         * @tparam I The tuple sequence indeces.
         * @tparam T The tuple's element members types.
         * @param t The resulting concatenated tuple.
         * @return The resulting concatenated tuple.
         */
        template <size_t ...I, typename ...T>
        __host__ __device__ inline constexpr auto concat(const tuple<indexer<I...>, T...>& t) noexcept
        -> tuple<T...>
        {
            return t;
        }

        /**
         * Applies a functor to all of a tuple's elements.
         * @tparam F The functor type to apply.
         * @tparam A The types of extra arguments.
         * @tparam I The tuple's indeces.
         * @tparam T The tuple's types.
         * @param lambda The functor to apply to the tuple.
         * @param t The tuple to apply functor to.
         * @return The new transformed tuple.
         */
        template <typename F, typename ...A, size_t ...I, typename ...T>
        __host__ __device__ inline constexpr auto apply(
            F&& lambda
          , const tuple<indexer<I...>, T...>& t
          , A&&... args
        ) -> tuple<decltype(lambda(impl::get<I>(t), std::forward<decltype(args)>(args)...))...>
        {
            return {lambda(impl::get<I>(t), std::forward<decltype(args)>(args)...)...};
        }

        /**
         * Performs a left-fold, a reduction operation on the given tuple.
         * @tparam F The functor type to combine the values with.
         * @tparam B The folding base and result value type.
         * @tparam I The tuple sequence indeces.
         * @tparam T The tuple's element members types.
         * @param lambda The functor used to created the new elements.
         * @param base The folding base value.
         * @param t The tuple to fold.
         * @return The reduction resulting value.
         */
        template <typename F, typename B, size_t ...I, typename ...T>
        __host__ __device__ inline constexpr B foldl(F&& lambda, const B& base, const tuple<indexer<I...>, T...>& t)
        {
            return foldl(lambda, lambda(base, head(t)), tail(t));
        }

        /**
         * The recursion base for a left-fold reduction.
         * @tparam F The functor type to combine the values with.
         * @tparam B The folding base and result value type.
         * @param base The folding base value.
         * @return The reduction base value.
         */
        template <typename F, typename B>
        __host__ __device__ inline constexpr B foldl(F&&, const B& base, const tuple<indexer<>>&)
        {
            return base;
        }

        /**
         * Performs a right-fold, a reduction operation on the given tuple.
         * @tparam F The functor type to combine the values with.
         * @tparam B The folding base and result value type.
         * @tparam I The tuple sequence indeces.
         * @tparam T The tuple's element members types.
         * @param lambda The functor used to created the new elements.
         * @param base The folding base value.
         * @param t The tuple to fold.
         * @return The reduction resulting value.
         */
        template <typename F, typename B, size_t ...I, typename ...T>
        __host__ __device__ inline constexpr B foldr(F&& lambda, const B& base, const tuple<indexer<I...>, T...>& t)
        {
            return lambda(last(t), foldr(lambda, base, init(t)));
        }

        /**
         * The recursion base for a right-fold reduction.
         * @tparam F The functor type to combine the values with.
         * @tparam B The folding base and result value type.
         * @param base The folding base value.
         * @return The reduction base value.
         */
        template <typename F, typename B>
        __host__ __device__ inline constexpr B foldr(F&&, const B& base, const tuple<indexer<>>&)
        {
            return base;
        }

        /**
         * Performs a left-scan fold and returns all intermediate steps values.
         * @tparam F The functor type to combine the values with.
         * @tparam B The folding base and result value type.
         * @tparam I The tuple sequence indeces.
         * @tparam T The tuple's element members types.
         * @param lambda The functor used to created the new elements.
         * @param base The folding base value.
         * @param t The tuple to fold.
         * @return The resulting fold tuple.
         */
        template <typename F, typename B, size_t ...I, typename ...T>
        __host__ __device__ inline constexpr auto scanl(
            F&& lambda
          , const B& base
          , const tuple<indexer<I...>, T...>& t
        ) -> ntuple<B, sizeof...(I) + 1>
        {
            return {base, impl::get<I>(scanl(lambda, lambda(base, head(t)), tail(t)))...};
        }

        /**
         * The recursion base for a left-scan fold reduction.
         * @tparam F The functor type to combine the values with.
         * @tparam B The folding base and result value type.
         * @param base The folding base value.
         * @return The reduction base tuple.
         */
        template <typename F, typename B>
        __host__ __device__ inline constexpr auto scanl(F&&, const B& base, const tuple<indexer<>>&)
        -> tuple<B>
        {
            return {base};
        }

        /**
         * Performs a right-scan fold and returns all intermediate steps values.
         * @tparam F The functor type to combine the values with.
         * @tparam B The folding base and result value type.
         * @tparam I The tuple sequence indeces.
         * @tparam T The tuple's element members types.
         * @param lambda The functor used to created the new elements.
         * @param base The folding base value.
         * @param t The tuple to fold.
         * @return The resulting fold tuple.
         */
        template <typename F, typename B, size_t ...I, typename ...T>
        __host__ __device__ inline constexpr auto scanr(
            F&& lambda
          , const B& base
          , const tuple<indexer<I...>, T...>& t
        ) -> ntuple<B, sizeof...(I) + 1>
        {
            return {impl::get<I>(scanr(lambda, lambda(last(t), base), init(t)))..., base};
        }

        /**
         * The recursion base for a right-scan fold reduction.
         * @tparam F The functor type to combine the values with.
         * @tparam B The folding base and result value type.
         * @param base The folding base value.
         * @return The reduction base tuple.
         */
        template <typename F, typename B>
        __host__ __device__ inline constexpr auto scanr(F&&, const B& base, const tuple<indexer<>>&)
        -> tuple<B>
        {
            return {base};
        }

        /**
         * Zips two tuples together eventually creating a tuple of pairs with types
         * intercalated from the two original tuples.
         * @tparam I The tuple sequence indeces.
         * @tparam T The first tuple's element members types.
         * @tparam U The second tuple's element members types.
         * @param a The first tuple to zip.
         * @param b The second tuple to zip.
         * @return The resulting zipped tuple.
         */
        template <size_t ...I, typename ...T, typename ...U>
        __host__ __device__ inline constexpr auto zip(
            const tuple<indexer<I...>, T...>& a
          , const tuple<indexer<I...>, U...>& b
        ) -> tuple<pair<T, U>...>
        {
            return {pair<T, U> {impl::get<I>(a), impl::get<I>(b)}...};
        }

        /**
         * Zips two tuples together by combining paired elements with a given functor.
         * Therefore, the resulting tuple does not contain pairs, but each result.
         * @tparam F The functor type to combine the values with.
         * @tparam I The tuple sequence indeces.
         * @tparam T The first tuple's element members types.
         * @tparam U The second tuple's element members types.
         * @param lambda The functor used to combine the elements.
         * @param a The first tuple to zip.
         * @param b The second tuple to zip.
         * @return The resulting tuple.
         */
        template <typename F, size_t ...I, typename ...T, typename ...U>
        __host__ __device__ inline constexpr auto zipwith(
            F&& lambda
          , const tuple<indexer<I...>, T...>& a
          , const tuple<indexer<I...>, U...>& b
        ) -> tuple<decltype(lambda(std::declval<T>(), std::declval<U>()))...>
        {
            return {lambda(impl::get<I>(a), impl::get<I>(b))...};
        }
    }
}
