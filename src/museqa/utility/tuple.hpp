/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A general tuple type abstraction implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <cstdint>
#include <utility>

#include <museqa/environment.h>
#include <museqa/utility.hpp>

#if defined(MUSEQA_COMPILER_NVCC)
  #pragma push
  #pragma diag_suppress = unrecognized_gcc_pragma
#endif

#include <fmt/format.h>

#if defined(MUSEQA_COMPILER_NVCC)
  #pragma pop
#endif

namespace museqa::utility
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
    class tuple : public tuple<std::make_index_sequence<sizeof...(T)>, T...>
    {
        private:
            typedef std::make_index_sequence<sizeof...(T)> indexer_type;
            typedef tuple<indexer_type, T...> underlying_type;

        public:
            __host__ __device__ inline constexpr tuple() noexcept = default;
            __host__ __device__ inline constexpr tuple(const tuple&) = default;
            __host__ __device__ inline constexpr tuple(tuple&&) = default;

            /**
             * Creates a new tuple instance from a list of values.
             * @param value The list of concrete values to create the tuple with.
             */
            __host__ __device__ inline constexpr tuple(const T&... value)
              : underlying_type {value...}
            {}

            /**
             * Creates a new tuple instance from a list of values to move.
             * @param value The list of concrete values to create the tuple with.
             */
            __host__ __device__ inline constexpr tuple(T&&... value)
              : underlying_type {std::forward<decltype(value)>(value)...}
            {}

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

            using underlying_type::tuple;

            __host__ __device__ inline tuple& operator=(const tuple&) = default;
            __host__ __device__ inline tuple& operator=(tuple&&) = default;

            using underlying_type::operator=;
    };

    namespace detail
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
            typedef T element_type;
            element_type value {};

            __host__ __device__ inline constexpr leaf() noexcept = default;
            __host__ __device__ inline constexpr leaf(const leaf&) = default;
            __host__ __device__ inline constexpr leaf(leaf&&) = default;

            /**
             * Constructs a new tuple leaf.
             * @param value The value to be contained by the leaf.
             */
            __host__ __device__ inline constexpr leaf(const element_type& value)
              : value {value}
            {}

            /**
             * Constructs a new tuple leaf by moving a value.
             * @param value The value to be moved into the leaf.
             */
            __host__ __device__ inline constexpr leaf(element_type&& value)
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
            __host__ __device__ inline constexpr leaf(const U& value)
              : value {static_cast<element_type>(value)}
            {}

            /**
             * Constructs a new tuple leaf by copying a foreign leaf.
             * @tparam U A convertible foreign type for the leaf's value.
             * @param other The leaf to copy contents from.
             */
            template <size_t J, typename U>
            __host__ __device__ inline constexpr leaf(const leaf<J, U>& other)
              : leaf {other.value}
            {}

            /**
             * Constructs a new tuple leaf by moving a foreign leaf.
             * @tparam U A convertible foreign type for the leaf's value.
             * @param other The leaf to move the contents from.
             */
            template <size_t J, typename U>
            __host__ __device__ inline constexpr leaf(leaf<J, U>&& other)
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
            typedef T& element_type;
            element_type value;

            __host__ __device__ inline constexpr leaf() noexcept = delete;
            __host__ __device__ inline constexpr leaf(const leaf&) noexcept = default;
            __host__ __device__ inline constexpr leaf(leaf&&) noexcept = default;

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
        __host__ __device__ inline constexpr T& get(leaf<I, T>& leaf) noexcept
        {
            return leaf.value;
        }

        /**
         * Retrieves the requested const-qualified tuple leaf and returns its value.
         * @tparam I The requested leaf index.
         * @tparam T The type of the requested leaf member.
         * @param leaf The selected const-qualified tuple leaf member.
         * @return The const-qualified leaf's value.
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
        __host__ __device__ constexpr auto repeater(std::index_sequence<I...>) noexcept
        -> tuple<typename identity<T, I>::type...>;

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
    class tuple<std::index_sequence<I...>, T...> : public detail::leaf<I, T>...
    {
        private:
            typedef std::index_sequence<I...> indexer_type;

        public:
            /**
             * Retrieves the type of a specific tuple element by its index.
             * @tparam I The requested element index.
             * @since 1.0
             */
            template <size_t J>
            using element = decltype(detail::type<J>(std::declval<tuple>()));

        public:
            static constexpr size_t count = sizeof...(I);

        public:
            __host__ __device__ inline constexpr tuple() noexcept = default;
            __host__ __device__ inline constexpr tuple(const tuple&) = default;
            __host__ __device__ inline constexpr tuple(tuple&&) = default;

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
              : detail::leaf<I, T> (std::forward<decltype(value)>(value))...
            {}

            /**
             * Creates a new tuple instance from a tuple of foreign types.
             * @tparam U The types of foreign tuple instance to copy from.
             * @param other The tuple which values must be copied from.
             */
            template <typename ...U>
            __host__ __device__ inline constexpr tuple(const tuple<indexer_type, U...>& other)
              : detail::leaf<I, T> (static_cast<detail::leaf<I, U>>(other))...
            {}

            /**
             * Creates a new tuple instance by moving a tuple of foreign types.
             * @tparam U The types of foreign tuple instance to move from.
             * @param other The tuple which values must be moved from.
             */
            template <typename ...U>
            __host__ __device__ inline constexpr tuple(tuple<indexer_type, U...>&& other)
              : detail::leaf<I, T> (std::forward<detail::leaf<I, U>>(other))...
            {}

            /**
             * Copies the values from a different tuple instance.
             * @param other The tuple the values must be copied from.
             * @return The current tuple instance.
             */
            __host__ __device__ inline tuple& operator=(const tuple& other)
            {
                return swallow(*this, detail::set<I>(*this, detail::get<I>(other))...);
            }

            /**
             * Moves the values from a different tuple instance.
             * @param other The tuple the values must be moved from.
             * @return The current tuple instance.
             */
            __host__ __device__ inline tuple& operator=(tuple&& other)
            {
                return swallow(*this, detail::set<I>(*this, detail::get<I>(std::forward<decltype(other)>(other)))...);
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
                return swallow(*this, detail::set<I>(*this, detail::get<I>(other))...);
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
                return swallow(*this, detail::set<I>(*this, detail::get<I>(std::forward<decltype(other)>(other)))...);
            }

            /**
             * Retrieves the value of a tuple member by its index.
             * @tparam J The requested member's index.
             * @return The member's value.
             */
            template <size_t J>
            __host__ __device__ inline constexpr auto get() noexcept -> decltype(auto)
            {
                return detail::get<J>(*this);
            }

            /**
             * Retrieves the value of a const-qualified tuple member by its index.
             * @tparam J The requested member's index.
             * @return The const-qualified member's value.
             */
            template <size_t J>
            __host__ __device__ inline constexpr auto get() const noexcept -> decltype(auto)
            {
                return detail::get<J>(*this);
            }

            /**
             * Updates the value of a tuple member by its index.
             * @tparam J The requested member's index.
             * @tparam U The member's new value's type.
             */
            template <size_t J, typename U>
            __host__ __device__ inline void set(U&& value)
            {
                detail::set<J>(*this, std::forward<decltype(value)>(value));
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
    class ntuple : public decltype(detail::repeater<T>(std::make_index_sequence<N>()))
    {
        private:
            typedef std::make_index_sequence<N> indexer_type;
            typedef decltype(detail::repeater<T>(indexer_type())) underlying_type;

        public:
            using element_type = T;

        public:
            __host__ __device__ inline constexpr ntuple() noexcept = default;
            __host__ __device__ inline constexpr ntuple(const ntuple&) = default;
            __host__ __device__ inline constexpr ntuple(ntuple&&) = default;

            /**
             * Creates a new tuple from a raw array.
             * @tparam U The array's type to create tuple from.
             * @param array The array to initialize the tuple's values from.
             */
            __host__ __device__ inline constexpr ntuple(const T* array)
              : ntuple {indexer_type(), array}
            {}

            /**
             * Creates a new tuple by moving a raw array.
             * @tparam U The array's type to create tuple from.
             * @param array The array to move into the tuple's values.
             */
            __host__ __device__ inline constexpr ntuple(T (&&array)[N])
              : ntuple {indexer_type(), std::forward<decltype(array)>(array)}
            {}

            /**
             * Creates a new tuple from a raw foreign array.
             * @tparam U The foreign array's type to create tuple from.
             * @param array The array to initialize the tuple's values from.
             */
            template <typename U>
            __host__ __device__ inline constexpr ntuple(const U* array)
              : ntuple {indexer_type(), array}
            {}

            /**
             * Creates a new tuple by moving a raw foreign array.
             * @tparam U The foreign array's type to create tuple from.
             * @param array The array to move into the tuple's values.
             */
            template <typename U>
            __host__ __device__ inline constexpr ntuple(U (&&array)[N])
              : ntuple {indexer_type(), std::forward<decltype(array)>(array)}
            {}

            using underlying_type::tuple;

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
            __host__ __device__ inline constexpr ntuple(std::index_sequence<I...>, const U* array)
              : underlying_type {array[I]...}
            {}

            /**
             * Creates a new tuple by moving the contents of an array.
             * @tparam U The foreign array type to create tuple from.
             * @tparam I The tuple's sequence index for inlining the array.
             * @param array The array to be moved.
             */
            template <typename U, size_t ...I>
            __host__ __device__ inline constexpr ntuple(std::index_sequence<I...>, U (&&array)[N])
              : underlying_type {std::forward<U>(array[I])...}
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
            __host__ __device__ inline constexpr pair(const pair&) = default;
            __host__ __device__ inline constexpr pair(pair&&) = default;

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
                return detail::get<0>(*this);
            }

            /**
             * Retrieves the second element of the pair.
             * @return The pair's second element's reference.
             */
            __host__ __device__ inline constexpr auto second() const noexcept -> const U&
            {
                return detail::get<1>(*this);
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

    /*
     * Deduction guides for generic tuple types.
     * @since 1.0
     */
    template <typename ...T> tuple(T...) -> tuple<T...>;
    template <typename T, size_t N> ntuple(T(&)[N]) -> ntuple<T, N>;
    template <typename T, typename U> pair(T, U) -> pair<T, U>;

    /**
     * Gathers variables references into a tuple instance, allowing them to capture
     * values directly from value tuples.
     * @tparam T The gathered variables types.
     * @param ref The gathered variables references.
     * @return The new tuple of references.
     */
    template <typename ...T>
    __host__ __device__ inline constexpr auto tie(T&... ref) noexcept -> tuple<T&...>
    {
        return {ref...};
    }

    /**
     * Gathers an array's elements' references into a tuple instance, allowing them
     * to capture values directly from value tuples.
     * @tparam T The gathered variables types.
     * @tparam N When an array, the size must be fixed.
     * @param ref The gathered variables references.
     * @return The new tuple of references.
     */
    template <typename T, size_t N>
    __host__ __device__ inline constexpr auto tie(T (&ref)[N]) noexcept -> ntuple<T&, N>
    {
        return {ref};
    }

    /**
     * Retrieves and returns the value of the first leaf of a tuple.
     * @tparam I The tuple sequence indeces to match from tuple.
     * @tparam T The list of tuple's element members types.
     * @param t The tuple to get the first element from.
     * @return The head value of tuple.
     */
    template <size_t ...I, typename ...T>
    __host__ __device__ inline constexpr auto head(const tuple<std::index_sequence<I...>, T...>& t) noexcept
    {
        return detail::get<0>(t);
    }

    /**
     * Retrieves and returns the value of the last leaf of a tuple.
     * @tparam I The tuple sequence indeces to match from tuple.
     * @tparam T The list of tuple's element members types.
     * @param t The tuple to get the last element from.
     * @return The last value of tuple.
     */
    template <size_t ...I, typename ...T>
    __host__ __device__ inline constexpr auto last(const tuple<std::index_sequence<I...>, T...>& t) noexcept
    {
        return detail::get<sizeof...(T) - 1>(t);
    }

    /**
     * Returns a tuple with its last leaf removed.
     * @tparam I The tuple sequence indeces to match from tuple.
     * @tparam T The list of tuple's element members types.
     * @param t The tuple to have its last element removed.
     * @return The new tuple with removed end.
     */
    template <size_t ...I, typename ...T>
    __host__ __device__ inline constexpr auto init(const tuple<std::index_sequence<0, I...>, T...>& t)
    {
        return tuple {detail::get<I - 1>(t)...};
    }

    /**
     * Returns a moved tuple with its last leaf removed.
     * @tparam I The tuple sequence indeces to match from tuple.
     * @tparam T The list of tuple's element members types.
     * @param t The tuple to have its last element removed.
     * @return The new moved tuple with removed end.
     */
    template <size_t ...I, typename ...T>
    __host__ __device__ inline constexpr auto init(tuple<std::index_sequence<0, I...>, T...>&& t)
    {
        return tuple {detail::get<I - 1>(std::forward<decltype(t)>(t))...};
    }

    /**
     * Returns a tuple with its first leaf removed.
     * @tparam I The tuple sequence indeces to match from tuple.
     * @tparam T The list of tuple's element members types.
     * @param t The tuple to have its first element removed.
     * @return The new tuple with removed head.
     */
    template <size_t ...I, typename ...T>
    __host__ __device__ inline constexpr auto tail(const tuple<std::index_sequence<0, I...>, T...>& t)
    {
        return tuple {detail::get<I>(t)...};
    }

    /**
     * Returns a moved tuple with its first leaf removed.
     * @tparam I The tuple sequence indeces to match from tuple.
     * @tparam T The list of tuple's element members types.
     * @param t The tuple to have its first element removed.
     * @return The new moved tuple with removed head.
     */
    template <size_t ...I, typename ...T>
    __host__ __device__ inline constexpr auto tail(tuple<std::index_sequence<0, I...>, T...>&& t)
    {
        return tuple {detail::get<I>(std::forward<decltype(t)>(t))...};
    }

    /**
     * The recursion base for a tuple concatenation.
     * @tparam I The tuple sequence indeces.
     * @tparam T The tuple's element members types.
     * @param t The resulting concatenated tuple.
     * @return The resulting concatenated tuple.
     */
    template <size_t ...I, typename ...T>
    __host__ __device__ inline constexpr auto concat(const tuple<std::index_sequence<I...>, T...>& t) noexcept
    {
        return t;
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
        const tuple<std::index_sequence<I...>, T...>& a
      , const tuple<std::index_sequence<J...>, U...>& b
      , const R&... tail
    ) {
        return concat(tuple {detail::get<I>(a)..., detail::get<J>(b)...}, tail...);
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
      , const tuple<std::index_sequence<I...>, T...>& t
      , A&&... args
    ) {
        return tuple {lambda(detail::get<I>(t), std::forward<decltype(args)>(args)...)...};
    }

    /**
     * The recursion base for a left-fold reduction.
     * @tparam F The functor type to combine the values with.
     * @tparam B The folding base and result value type.
     * @param base The folding base value.
     * @return The reduction base value.
     */
    template <typename F, typename B>
    __host__ __device__ inline constexpr B foldl(F&&, const B& base, const tuple<std::index_sequence<>>&)
    {
        return base;
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
    __host__ __device__ inline constexpr B foldl(
        F&& lambda
      , const B& base
      , const tuple<std::index_sequence<I...>, T...>& t
    ) {
        return foldl(lambda, lambda(base, head(t)), tail(t));
    }

    /**
     * The recursion base for a right-fold reduction.
     * @tparam F The functor type to combine the values with.
     * @tparam B The folding base and result value type.
     * @param base The folding base value.
     * @return The reduction base value.
     */
    template <typename F, typename B>
    __host__ __device__ inline constexpr B foldr(F&&, const B& base, const tuple<std::index_sequence<>>&)
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
    __host__ __device__ inline constexpr B foldr(
        F&& lambda
      , const B& base
      , const tuple<std::index_sequence<I...>, T...>& t
    ) {
        return lambda(last(t), foldr(lambda, base, init(t)));
    }

    /**
     * The recursion base for a left-scan fold reduction.
     * @tparam F The functor type to combine the values with.
     * @tparam B The folding base and result value type.
     * @param base The folding base value.
     * @return The reduction base tuple.
     */
    template <typename F, typename B>
    __host__ __device__ inline constexpr auto scanl(F&&, const B& base, const tuple<std::index_sequence<>>&)
    {
        return tuple {base};
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
      , const tuple<std::index_sequence<I...>, T...>& t
    ) {
        return tuple {base, detail::get<I>(scanl(lambda, lambda(base, head(t)), tail(t)))...};
    }

    /**
     * The recursion base for a right-scan fold reduction.
     * @tparam F The functor type to combine the values with.
     * @tparam B The folding base and result value type.
     * @param base The folding base value.
     * @return The reduction base tuple.
     */
    template <typename F, typename B>
    __host__ __device__ inline constexpr auto scanr(F&&, const B& base, const tuple<std::index_sequence<>>&)
    {
        return tuple {base};
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
      , const tuple<std::index_sequence<I...>, T...>& t
    ) {
        return tuple {detail::get<I>(scanr(lambda, lambda(last(t), base), init(t)))..., base};
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
        const tuple<std::index_sequence<I...>, T...>& a
      , const tuple<std::index_sequence<I...>, U...>& b
    ) {
        return tuple {pair(detail::get<I>(a), detail::get<I>(b))...};
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
      , const tuple<std::index_sequence<I...>, T...>& a
      , const tuple<std::index_sequence<I...>, U...>& b
    ) {
        return tuple {lambda(detail::get<I>(a), detail::get<I>(b))...};
    }
}

/**
 * Implements a string formatter for a generic tuple type, thus allowing tuples
 * to be printed whenever its contents are also printable.
 * @tparam T The list of tuple's element types.
 * @since 1.0
 */
template <typename ...T>
class fmt::formatter<museqa::utility::tuple<T...>>
{
    private:
        typedef museqa::utility::tuple<T...> target_type;
        static constexpr size_t count = target_type::count;

    public:
        /**
         * Evaluates the formatter's parsing context.
         * @tparam C The parsing context type.
         * @param ctx The parsing context instance.
         * @return The processed and evaluated parsing context.
         */
        template <typename C>
        inline constexpr auto parse(C& ctx) const
        {
            return ctx.begin();
        }

        /**
         * Formats the tuple into a printable string.
         * @tparam F The formatting context type.
         * @param tuple The tuple to be formatted into a string.
         * @param ctx The formatting context instance.
         * @return The formatting context instance.
         */
        template <size_t ...I, typename F>
        auto format(const museqa::utility::tuple<std::index_sequence<I...>, T...>& tuple, F& ctx) const
        {
            std::string args[] = {fmt::format("{}", tuple.template get<I>())...};
            return fmt::format_to(ctx.out(), "({})", fmt::join(args, args + count, ", "));
        }
};

/**
 * Implements a string formatter for a generic pair-tuple type, which will be printed
 * as a regular tuple with 2 elements of possibly distinct types.
 * @tparam T The pair's first element type.
 * @tparam U The pair's second element type.
 * @since 1.0
 */
template <typename T, typename U>
struct fmt::formatter<museqa::utility::pair<T, U>> : fmt::formatter<museqa::utility::tuple<T, U>>
{};

/**
 * Implements a string formatter for a generic n-tuple type, which will be printed
 * as a regular tuple with many elements of the same type.
 * @tparam T The tuple's elements type.
 * @tparam N The number of elements in tuple.
 * @since 1.0
 */
template <typename T, size_t N>
struct fmt::formatter<museqa::utility::ntuple<T, N>> : fmt::formatter<
    decltype(museqa::utility::detail::repeater<T>(std::make_index_sequence<N>()))
> {};

template <typename ...T>
struct std::tuple_size<museqa::utility::tuple<T...>>
  : std::integral_constant<size_t, museqa::utility::tuple<T...>::count>
{};

template <typename T, typename U>
struct std::tuple_size<museqa::utility::pair<T, U>> : std::tuple_size<museqa::utility::tuple<T, U>>
{};

template <typename T, size_t N>
struct std::tuple_size<museqa::utility::ntuple<T, N>> : std::tuple_size<
    decltype(museqa::utility::detail::repeater<T>(std::make_index_sequence<N>()))
> {};

template <size_t I, typename ...T>
struct std::tuple_element<I, museqa::utility::tuple<T...>> : museqa::identity<
    decltype(museqa::utility::detail::get<I>(std::declval<museqa::utility::tuple<T...>>()))
  , 0
> {};

template <size_t I, typename T, typename U>
struct std::tuple_element<I, museqa::utility::pair<T, U>> : tuple_element<I, museqa::utility::tuple<T, U>>
{};

template <size_t I, typename T, size_t N>
struct std::tuple_element<I, museqa::utility::ntuple<T, N>> : tuple_element<
    I
  , decltype(museqa::utility::detail::repeater<T>(std::make_index_sequence<N>()))
> {};
