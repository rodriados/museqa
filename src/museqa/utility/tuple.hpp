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
#include <museqa/utility/detail/lambda.hpp>

MUSEQA_BEGIN_NAMESPACE

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
    class tuple_t : public tuple_t<identity_t<std::make_index_sequence<sizeof...(T)>>, T...>
    {
        public:
            static constexpr size_t count = sizeof...(T);

        private:
            typedef museqa::identity_t<std::make_index_sequence<count>> identity_t;
            typedef tuple_t<identity_t, T...> underlying_t;

        public:
            using underlying_t::tuple_t;
            using underlying_t::operator=;
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
        struct leaf_t
        {
            typedef T element_t;
            element_t value;

            __host__ __device__ inline constexpr leaf_t() = default;
            __host__ __device__ inline constexpr leaf_t(const leaf_t&) = default;
            __host__ __device__ inline constexpr leaf_t(leaf_t&&) = default;

            /**
             * Constructs a new tuple leaf.
             * @param value The value to be contained by the leaf.
             */
            __host__ __device__ inline constexpr leaf_t(const element_t& value)
              : value (value)
            {}

            /**
             * Constructs a new tuple leaf by moving a foreign value.
             * @tparam U The foreign value's type to be possibly moved.
             * @param value The foreign value to be moved into the leaf.
             */
            template <typename U>
            __host__ __device__ inline constexpr leaf_t(U&& value)
              : value (std::forward<decltype(value)>(value))
            {}

            /**
             * Constructs a new leaf by copying from a foreign tuple's leaf.
             * @tparam U The foreign tuple leaf's element type.
             * @param other The leaf to copy from.
             */
            template <typename U>
            __host__ __device__ inline constexpr leaf_t(const leaf_t<I, U>& other)
              : value (other.value)
            {}

            /**
             * Constructs a new leaf by moving from a foreign tuple's leaf.
             * @tparam U The foreign tuple leaf's element type.
             * @param other The leaf to move from.
             */
            template <typename U>
            __host__ __device__ inline constexpr leaf_t(leaf_t<I, U>&& other)
              : value (std::forward<decltype(other.value)>(other.value))
            {}

            __host__ __device__ inline leaf_t& operator=(const leaf_t&) = default;
            __host__ __device__ inline leaf_t& operator=(leaf_t&&) = default;

            /**
             * Copies the contents of a foreign tuple's leaf.
             * @tparam U The foreign tuple leaf's element type.
             * @param other The leaf to copy from.
             * @return The current leaf instance.
             */
            template <typename U>
            __host__ __device__ inline leaf_t& operator=(const leaf_t<I, U>& other)
            {
                return swallow(*this, value = other.value);
            }

            /**
             * Moves the contents of a foreign tuple's leaf.
             * @tparam U The foreign tuple leaf's element type.
             * @param other The leaf to move from.
             * @return The current leaf instance.
             */
            template <typename U>
            __host__ __device__ inline leaf_t& operator=(leaf_t<I, U>&& other)
            {
                return swallow(*this, value = std::forward<decltype(other.value)>(other.value));
            }
        };

        /**
         * Retrieves the requested tuple leaf and returns its value.
         * @tparam I The requested leaf index.
         * @tparam T The type of the requested leaf member.
         * @param leaf The selected tuple leaf member.
         * @return The leaf's value.
         */
        template <size_t I, typename T>
        __host__ __device__ inline constexpr T& get(leaf_t<I, T>& leaf) noexcept
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
        __host__ __device__ inline constexpr const T& get(const leaf_t<I, T>& leaf) noexcept
        {
            return leaf.value;
        }

        /**
         * Retrieves the requested tuple leaf and moves its contents.
         * @tparam I The requested leaf index.
         * @tparam T The type of the requested leaf member.
         * @param leaf The selected tuple leaf member.
         * @return The leaf value's move reference.
         */
        template <size_t I, typename T>
        __host__ __device__ inline constexpr auto get(leaf_t<I, T>&& leaf) noexcept
        {
            return std::forward<decltype(leaf.value)>(leaf.value);
        }

        /**
         * Modifies a tuple leaf by moving a value into it.
         * @tparam I The requested leaf index.
         * @tparam T The type of requested leaf member.
         * @param leaf The selected tuple leaf member.
         * @param value The value to move into the leaf.
         */
        template <size_t I, typename T, typename U>
        __host__ __device__ inline void set(leaf_t<I, T>& leaf, U&& value)
        {
            leaf.value = std::forward<decltype(value)>(value);
        }

        /**
         * Creates a tuple with repeated types.
         * @tparam T The type to be repeated as tuple elements.
         * @tparam I The tuple's type index sequence.
         */
        template <typename T, size_t ...I>
        __host__ __device__ constexpr auto repeater(std::index_sequence<I...>) noexcept
        -> tuple_t<typename identity_t<T, I>::type...>;

        /**
         * Accesses the internal declared type of a tuple leaf.
         * @tparam I The index of the leaf to be accessed in the tuple.
         * @tparam T The extracted tuple element type.
         */
        template <size_t I, typename T>
        __host__ __device__ constexpr auto type(leaf_t<I, T>) noexcept
        -> typename leaf_t<I, T>::element_t;
    }

    /**
     * The base tuple type.
     * @tparam I The sequence indeces for the tuple elements' types.
     * @tparam T The list of tuple elements' types.
     * @since 1.0
     */
    template <size_t ...I, typename ...T>
    class tuple_t<identity_t<std::index_sequence<I...>>, T...> : public detail::leaf_t<I, T>...
    {
        private:
            typedef museqa::identity_t<std::index_sequence<I...>> identity_t;

        public:
            /**
             * Retrieves the type of a specific tuple element by its index.
             * @tparam J The requested element index.
             * @since 1.0
             */
            template <size_t J>
            using element_t = decltype(detail::type<J>(std::declval<tuple_t>()));

        public:
            __host__ __device__ inline constexpr tuple_t() = default;
            __host__ __device__ inline constexpr tuple_t(const tuple_t&) = default;
            __host__ __device__ inline constexpr tuple_t(tuple_t&&) = default;

            /**
             * Creates a new tuple instance from a list of foreign values.
             * @tparam U The foreign values' types to build the tuple from.
             * @param value The list of foreign values to create the tuple with.
             */
            template <
                typename ...U
              , typename std::enable_if<sizeof...(U) == sizeof...(T), int>::type = 0
            >
            __host__ __device__ inline constexpr tuple_t(U&&... value)
              : detail::leaf_t<I, T> (std::forward<decltype(value)>(value))...
            {}

            /**
             * Creates a new tuple instance from a tuple of foreign types.
             * @tparam U The types of foreign tuple instance to copy from.
             * @param other The foreign tuple which values must be copied from.
             */
            template <typename ...U>
            __host__ __device__ inline constexpr tuple_t(const tuple_t<identity_t, U...>& other)
              : detail::leaf_t<I, T> (static_cast<const detail::leaf_t<I, U>&>(other))...
            {}

            /**
             * Creates a new tuple instance by moving a tuple of foreign types.
             * @tparam U The types of foreign tuple instance to move from.
             * @param other The foreign tuple which values must be moved from.
             */
            template <typename ...U>
            __host__ __device__ inline constexpr tuple_t(tuple_t<identity_t, U...>&& other)
              : detail::leaf_t<I, T> (std::forward<detail::leaf_t<I, U>>(other))...
            {}

            __host__ __device__ inline tuple_t& operator=(const tuple_t&) = default;
            __host__ __device__ inline tuple_t& operator=(tuple_t&&) = default;

            /**
             * Copies the values from a foreign tuple instance.
             * @tparam U The types of foreign tuple instance to copy from.
             * @param other The tuple the values must be copied from.
             * @return The current tuple instance.
             */
            template <typename ...U>
            __host__ __device__ inline tuple_t& operator=(const tuple_t<identity_t, U...>& other)
            {
                return swallow(*this, detail::leaf_t<I, T>::operator=(other)...);
            }

            /**
             * Moves the values from a foreign tuple instance.
             * @tparam U The types of the foreign tuple instance to move from.
             * @param other The tuple the values must be moved from.
             * @return The current tuple instance.
             */
            template <typename ...U>
            __host__ __device__ inline tuple_t& operator=(tuple_t<identity_t, U...>&& other)
            {
                return swallow(*this, detail::leaf_t<I, T>::operator=(std::forward<decltype(other)>(other))...);
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
    class ntuple_t : public decltype(detail::repeater<T>(std::make_index_sequence<N>()))
    {
        private:
            typedef std::make_index_sequence<N> indexer_t;
            typedef decltype(detail::repeater<T>(indexer_t())) underlying_t;

        public:
            __host__ __device__ inline constexpr ntuple_t() noexcept = default;
            __host__ __device__ inline constexpr ntuple_t(const ntuple_t&) = default;
            __host__ __device__ inline constexpr ntuple_t(ntuple_t&&) = default;

            /**
             * Creates a new tuple from a raw foreign array.
             * @tparam U The foreign array's type to create tuple from.
             * @param array The array to initialize the tuple's values from.
             */
            template <
                typename U
              , typename = typename std::enable_if<
                    std::is_pointer<typename std::remove_reference<U>::type>() ||
                    std::is_array<typename std::remove_reference<U>::type>()
                >::type
            >
            __host__ __device__ inline constexpr ntuple_t(U&& array)
              : ntuple_t {indexer_t(), array}
            {}

            /**
             * Creates a new tuple by moving a raw foreign array.
             * @tparam U The foreign array's type to create tuple from.
             * @param array The array to move into the tuple's values.
             */
            template <typename U>
            __host__ __device__ inline constexpr ntuple_t(U (&&array)[N])
              : ntuple_t {indexer_t(), std::forward<decltype(array)>(array)}
            {}

            using underlying_t::tuple_t;

            __host__ __device__ inline ntuple_t& operator=(const ntuple_t&) = default;
            __host__ __device__ inline ntuple_t& operator=(ntuple_t&&) = default;

            using underlying_t::operator=;

        private:
            /**
             * Creates a new tuple by inlining an array.
             * @tparam U The foreign array type to create tuple from.
             * @tparam I The tuple's sequence index for inlining the array.
             * @param array The array to inline.
             */
            template <typename U, size_t ...I>
            __host__ __device__ inline constexpr ntuple_t(std::index_sequence<I...>, U&& array)
              : underlying_t {array[I]...}
            {}

            /**
             * Creates a new tuple by moving the contents of an array.
             * @tparam U The foreign array type to create tuple from.
             * @tparam I The tuple's sequence index for inlining the array.
             * @param array The array to be moved.
             */
            template <typename U, size_t ...I>
            __host__ __device__ inline constexpr ntuple_t(std::index_sequence<I...>, U (&&array)[N])
              : underlying_t {std::move(array[I])...}
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
    class pair_t : public tuple_t<T, U>
    {
        private:
            typedef tuple_t<T, U> underlying_t;

        public:
            __host__ __device__ inline constexpr pair_t() noexcept = default;
            __host__ __device__ inline constexpr pair_t(const pair_t&) = default;
            __host__ __device__ inline constexpr pair_t(pair_t&&) = default;

            using underlying_t::tuple_t;

            __host__ __device__ inline pair_t& operator=(const pair_t&) = default;
            __host__ __device__ inline pair_t& operator=(pair_t&&) = default;

            using underlying_t::operator=;

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
    using tuple_element_t = typename T::template element_t<I>;

    /*
     * Deduction guides for generic tuple types.
     * @since 1.0
     */
    template <typename ...T> tuple_t(T...) -> tuple_t<T...>;
    template <typename T, typename U> pair_t(T, U) -> pair_t<T, U>;
    template <typename T, size_t N> ntuple_t(const T(&)[N]) -> ntuple_t<T, N>;
    template <typename T, size_t N> ntuple_t(T(&&)[N]) -> ntuple_t<T, N>;

    /**
     * Gathers variables references into a tuple instance, allowing them to capture
     * values directly from value tuples.
     * @tparam T The gathered variables types.
     * @param ref The gathered variables references.
     * @return The new tuple of references.
     */
    template <typename ...T>
    __host__ __device__ inline constexpr decltype(auto) tie(T&... ref) noexcept
    {
        return tuple_t<T&...>(ref...);
    }

    /**
     * Gathers an array's elements' references into a tuple instance, allowing them
     * to capture values directly from value tuples.
     * @tparam T The array's elements' type.
     * @tparam N The size of the given array.
     * @param ref The target array's reference.
     * @return The new tuple of references.
     */
    template <typename T, size_t N>
    __host__ __device__ inline constexpr decltype(auto) tie(T (&ref)[N]) noexcept
    {
        return ntuple_t<T&, N>(ref);
    }

    /**
     * Gathers move-references from an array's elements into a tuple instance, allowing
     * them to be moved directly into other variables.
     * @tparam T The array's elements' type.
     * @tparam N The size of the given array.
     * @param ref The target array's move-reference.
     * @return The new tuple of move-references.
     */
    template <typename T, size_t N>
    __host__ __device__ inline constexpr decltype(auto) tie(T (&&ref)[N]) noexcept
    {
        return ntuple_t<T&&, N>(std::forward<decltype(ref)>(ref));
    }

    /**
     * Retrieves and returns the value of the first leaf of a tuple.
     * @tparam I The tuple sequence indeces to match from tuple.
     * @tparam T The list of tuple's element members types.
     * @param t The tuple to get the first element from.
     * @return The head value of tuple.
     */
    template <size_t ...I, typename ...T>
    __host__ __device__ inline constexpr decltype(auto) head(
        const tuple_t<identity_t<std::index_sequence<I...>>, T...>& t
    ) noexcept {
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
    __host__ __device__ inline constexpr decltype(auto) last(
        const tuple_t<identity_t<std::index_sequence<I...>>, T...>& t
    ) noexcept {
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
    __host__ __device__ inline constexpr decltype(auto) init(
        const tuple_t<identity_t<std::index_sequence<0, I...>>, T...>& t
    ) {
        return tuple_t<tuple_element_t<tuple_t<T...>, I-1>...>(detail::get<I-1>(t)...);
    }

    /**
     * Returns a tuple with its first leaf removed.
     * @tparam I The tuple sequence indeces to match from tuple.
     * @tparam T The list of tuple's element members types.
     * @param t The tuple to have its first element removed.
     * @return The new tuple with removed head.
     */
    template <size_t ...I, typename ...T>
    __host__ __device__ inline constexpr decltype(auto) tail(
        const tuple_t<identity_t<std::index_sequence<0, I...>>, T...>& t
    ) {
        return tuple_t<tuple_element_t<tuple_t<T...>, I>...>(detail::get<I>(t)...);
    }

    /**
     * Appends an elements to the end of a tuple.
     * @tparam E The type of the element to append to tuple.
     * @tparam I The tuple sequence indeces to match from tuple.
     * @tparam T The list of tuple's element members types.
     * @param t The tuple to have an element appended to.
     * @param element The element to append to the tuple.
     * @return The resulting tuple.
     */
    template <typename E, size_t ...I, typename ...T>
    __host__ __device__ inline constexpr decltype(auto) append(
        const tuple_t<identity_t<std::index_sequence<I...>>, T...>& t
      , E&& element
    ) {
        return tuple_t<T..., E>(detail::get<I>(t)..., std::forward<decltype(element)>(element));
    }

    /**
     * Prepends an elements to the end of a tuple.
     * @tparam E The type of the element to prepend to tuple.
     * @tparam I The tuple sequence indeces to match from tuple.
     * @tparam T The list of tuple's element members types.
     * @param t The tuple to have an element prepended to.
     * @param element The element to prepend to the tuple.
     * @return The resulting tuple.
     */
    template <typename E, size_t ...I, typename ...T>
    __host__ __device__ inline constexpr decltype(auto) prepend(
        const tuple_t<identity_t<std::index_sequence<I...>>, T...>& t
      , E&& element
    ) {
        return tuple_t<E, T...>(std::forward<decltype(element)>(element), detail::get<I>(t)...);
    }

    /**
     * The recursion base for a tuple concatenation.
     * @tparam I The tuple sequence indeces.
     * @tparam T The tuple's element members types.
     * @param t The resulting concatenated tuple.
     * @return The resulting concatenated tuple.
     */
    template <size_t ...I, typename ...T>
    __host__ __device__ inline constexpr decltype(auto) concat(
        const tuple_t<identity_t<std::index_sequence<I...>>, T...>& t
    ) {
        return tuple_t<T...>(detail::get<I>(t)...);
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
    __host__ __device__ inline constexpr decltype(auto) concat(
        const tuple_t<identity_t<std::index_sequence<I...>>, T...>& a
      , const tuple_t<identity_t<std::index_sequence<J...>>, U...>& b
      , const R&... tail
    ) {
        return concat(tuple_t<T..., U...>(detail::get<I>(a)..., detail::get<J>(b)...), tail...);
    }

    /**
     * Reverses the tuple.
     * @tparam I The tuple sequence indeces.
     * @tparam T The tuple's element members types.
     * @param t The tuple to be reversed.
     * @return The reversed tuple.
     */
    template <size_t ...I, typename ...T>
    __host__ __device__ inline constexpr decltype(auto) reverse(
        const tuple_t<identity_t<std::index_sequence<I...>>, T...>& t
    ) {
        constexpr size_t J = sizeof...(I);
        return tuple_t<tuple_element_t<tuple_t<T...>, J-I-1>...>(detail::get<J-I-1>(t)...);
    }

    /**
     * Applies a functor to all of a tuple's elements.
     * @tparam F The functor type to apply.
     * @tparam A The types of extra arguments.
     * @tparam I The tuple's indeces.
     * @tparam T The tuple's types.
     * @param lambda The functor to apply to the tuple.
     * @param t The tuple to apply functor to.
     * @param args The remaining functor arguments.
     * @return The new transformed tuple.
     */
    template <typename F, typename ...A, size_t ...I, typename ...T>
    __host__ __device__ inline constexpr decltype(auto) apply(
        F&& lambda
      , const tuple_t<identity_t<std::index_sequence<I...>>, T...>& t
      , A&&... args
    ) {
        return tuple_t(detail::polymorphic_call(lambda, detail::get<I>(t), args...)...);
    }

    /**
     * Iterates over a tuple's elements using a functor.
     * @tparam F The functor type to apply.
     * @tparam A The types of extra arguments.
     * @tparam I The tuple's indeces.
     * @tparam T The tuple's types.
     * @param lambda The functor to apply to the tuple.
     * @param t The tuple to iterate over.
     * @param args The remaining functor arguments.
     */
    template <typename F, typename ...A, size_t ...I, typename ...T>
    __host__ __device__ inline constexpr void foreach(
        F&& lambda
      , const tuple_t<identity_t<std::index_sequence<I...>>, T...>& t
      , A&&... args
    ) {
        if constexpr (sizeof...(I) > 0) {
            detail::polymorphic_call(lambda, head(t), args...);
            foreach(lambda, tail(t), args...);
        }
    }

    /**
     * Iterates over a tuple's elements, in reverse, using a functor.
     * @tparam F The functor type to apply.
     * @tparam A The types of extra arguments.
     * @tparam I The tuple's indeces.
     * @tparam T The tuple's types.
     * @param lambda The functor to apply to the tuple.
     * @param t The tuple to iterate over.
     * @param args The remaining functor arguments.
     */
    template <typename F, typename ...A, size_t ...I, typename ...T>
    __host__ __device__ inline constexpr void rforeach(
        F&& lambda
      , const tuple_t<identity_t<std::index_sequence<I...>>, T...>& t
      , A&&... args
    ) {
        if constexpr (sizeof...(I) > 0) {
            detail::polymorphic_call(lambda, last(t), args...);
            rforeach(lambda, init(t), args...);
        }
    }

    /**
     * Performs a left-fold, a reduction operation on the given tuple.
     * @tparam F The functor type to combine the values with.
     * @tparam B The folding base and result value type.
     * @tparam I The tuple sequence indeces.
     * @tparam T The tuple's element members types.
     * @param lambda The functor used to create the new elements.
     * @param base The folding base value.
     * @param t The tuple to fold.
     * @return The reduction resulting value.
     */
    template <typename F, typename B, size_t ...I, typename ...T>
    __host__ __device__ inline constexpr decltype(auto) foldl(
        F&& lambda
      , B&& base
      , const tuple_t<identity_t<std::index_sequence<I...>>, T...>& t
    ) {
        if constexpr (sizeof...(I) > 0) {
            return foldl(lambda, detail::polymorphic_call(lambda, base, head(t)), tail(t));
        } else {
            return base;
        }
    }

    /**
     * Performs a left-fold operation without a folding base.
     * @tparam F The functor type to combine the values with.
     * @tparam I The tuple sequence indeces.
     * @tparam T The tuple's element members types.
     * @param lambda The functor used to create the new elements.
     * @param t The tuple to fold.
     * @return The reduction resulting value.
     */
    template <typename F, size_t ...I, typename ...T>
    __host__ __device__ inline constexpr decltype(auto) foldl1(
        F&& lambda
      , const tuple_t<identity_t<std::index_sequence<0, I...>>, T...>& t
    ) {
        return foldl(lambda, head(t), tail(t));
    }

    /**
     * Performs a right-fold, a reduction operation on the given tuple.
     * @tparam F The functor type to combine the values with.
     * @tparam B The folding base and result value type.
     * @tparam I The tuple sequence indeces.
     * @tparam T The tuple's element members types.
     * @param lambda The functor used to create the new elements.
     * @param base The folding base value.
     * @param t The tuple to fold.
     * @return The reduction resulting value.
     */
    template <typename F, typename B, size_t ...I, typename ...T>
    __host__ __device__ inline constexpr decltype(auto) foldr(
        F&& lambda
      , B&& base
      , const tuple_t<identity_t<std::index_sequence<I...>>, T...>& t
    ) {
        if constexpr (sizeof...(I) > 0) {
            return foldr(lambda, detail::polymorphic_call(lambda, base, last(t)), init(t));
        } else {
            return base;
        }
    }

    /**
     * Performs a right-fold operation without a folding base.
     * @tparam F The functor type to combine the values with.
     * @tparam I The tuple sequence indeces.
     * @tparam T The tuple's element members types.
     * @param lambda The functor used to create the new elements.
     * @param t The tuple to fold.
     * @return The reduction resulting value.
     */
    template <typename F, size_t ...I, typename ...T>
    __host__ __device__ inline constexpr decltype(auto) foldr1(
        F&& lambda
      , const tuple_t<identity_t<std::index_sequence<0, I...>>, T...>& t
    ) {
        return foldr(lambda, last(t), init(t));
    }

    /**
     * Performs a left-scan fold and returns all intermediate steps values.
     * @tparam F The functor type to combine the values with.
     * @tparam B The folding base and result value type.
     * @tparam I The tuple sequence indeces.
     * @tparam T The tuple's element members types.
     * @param lambda The functor used to create the new elements.
     * @param base The folding base value.
     * @param t The tuple to fold.
     * @return The resulting fold tuple.
     */
    template <typename F, typename B, size_t ...I, typename ...T>
    __host__ __device__ inline constexpr decltype(auto) scanl(
        F&& lambda
      , B&& base
      , const tuple_t<identity_t<std::index_sequence<I...>>, T...>& t
    ) {
        if constexpr (sizeof...(I) > 0) {
            return prepend(
                scanl(lambda, detail::polymorphic_call(lambda, base, head(t)), tail(t))
              , std::forward<decltype(base)>(base)
            );
        } else {
            return tuple_t(base);
        }
    }

    /**
     * Performs a left-scan operation without a folding base.
     * @tparam F The functor type to combine the values with.
     * @tparam I The tuple sequence indeces.
     * @tparam T The tuple's element members types.
     * @param lambda The functor used to create the new elements.
     * @param t The tuple to fold.
     * @return The resulting fold tuple.
     */
    template <typename F, size_t ...I, typename ...T>
    __host__ __device__ inline constexpr decltype(auto) scanl1(
        F&& lambda
      , const tuple_t<identity_t<std::index_sequence<0, I...>>, T...>& t
    ) {
        return scanl(lambda, head(t), tail(t));
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
    __host__ __device__ inline constexpr decltype(auto) scanr(
        F&& lambda
      , B&& base
      , const tuple_t<identity_t<std::index_sequence<I...>>, T...>& t
    ) {
        if constexpr (sizeof...(I) > 0) {
            return append(
                scanr(lambda, detail::polymorphic_call(lambda, base, last(t)), init(t))
              , std::forward<decltype(base)>(base)
            );
        } else {
            return tuple_t(base);
        }
    }

    /**
     * Performs a right-scan operation without a folding base.
     * @tparam F The functor type to combine the values with.
     * @tparam I The tuple sequence indeces.
     * @tparam T The tuple's element members types.
     * @param lambda The functor used to create the new elements.
     * @param t The tuple to fold.
     * @return The resulting fold tuple.
     */
    template <typename F, size_t ...I, typename ...T>
    __host__ __device__ inline constexpr decltype(auto) scanr1(
        F&& lambda
      , const tuple_t<identity_t<std::index_sequence<0, I...>>, T...>& t
    ) {
        return scanr(lambda, last(t), init(t));
    }

    /**
     * Zips two tuples together eventually creating a tuple of pairs with types
     * intercalated from the two original tuples.
     * @tparam I The tuples' sequence indeces.
     * @tparam T The first tuple's element members types.
     * @tparam U The second tuple's element members types.
     * @param a The first tuple to zip.
     * @param b The second tuple to zip.
     * @return The resulting zipped tuple.
     */
    template <size_t ...I, typename ...T, typename ...U>
    __host__ __device__ inline constexpr decltype(auto) zip(
        const tuple_t<identity_t<std::index_sequence<I...>>, T...>& a
      , const tuple_t<identity_t<std::index_sequence<I...>>, U...>& b
    ) {
        return tuple_t(pair_t<T, U>(detail::get<I>(a), detail::get<I>(b))...);
    }

    /**
     * Zips two tuples together by combining paired elements with a given functor.
     * Therefore, the resulting tuple does not contain pairs, but each result.
     * @tparam F The functor type to combine the values with.
     * @tparam I The tuples' sequence indeces.
     * @tparam T The first tuple's element members types.
     * @tparam U The second tuple's element members types.
     * @param lambda The functor used to combine the elements.
     * @param a The first tuple to zip.
     * @param b The second tuple to zip.
     * @return The resulting tuple.
     */
    template <typename F, size_t ...I, typename ...T, typename ...U>
    __host__ __device__ inline constexpr decltype(auto) zipwith(
        F&& lambda
      , const tuple_t<identity_t<std::index_sequence<I...>>, T...>& a
      , const tuple_t<identity_t<std::index_sequence<I...>>, U...>& b
    ) {
        return tuple_t(detail::polymorphic_call(lambda, detail::get<I>(a), detail::get<I>(b))...);
    }
}

MUSEQA_END_NAMESPACE

/**
 * Informs the size of a generic tuple, allowing it to be deconstructed.
 * @tparam T The tuple's elements' types.
 * @since 1.0
 */
template <typename ...T>
struct std::tuple_size<museqa::utility::tuple_t<T...>>
  : std::integral_constant<size_t, museqa::utility::tuple_t<T...>::count> {};

/**
 * Informs the size of a generic pair, allowing it to be deconstructed.
 * @tparam T The pair's first element type.
 * @tparam U The pair's second element type.
 * @since 1.0
 */
template <typename T, typename U>
struct std::tuple_size<museqa::utility::pair_t<T, U>>
  : std::tuple_size<museqa::utility::tuple_t<T, U>> {};

/**
 * Informs the size of a generic n-tuple, allowing it to be deconstructed.
 * @tparam T The n-tuple's elements' type.
 * @tparam N The total number of elements in the n-tuple.
 * @since 1.0
 */
template <typename T, size_t N>
struct std::tuple_size<museqa::utility::ntuple_t<T, N>>
  : std::integral_constant<size_t, N> {};

/**
 * Retrieves the deconstruction type of a tuple's element.
 * @tparam I The index of the requested tuple element.
 * @tparam T The tuple's elements' types.
 * @since 1.0
 */
template <size_t I, typename ...T>
struct std::tuple_element<I, museqa::utility::tuple_t<T...>>
  : museqa::identity_t<museqa::utility::tuple_element_t<museqa::utility::tuple_t<T...>, I>> {};

/**
 * Retrieves the deconstruction type of a pair's element.
 * @tparam I The index of the requested pair element.
 * @tparam T The pair's first element type.
 * @tparam U The pair's second element type.
 * @since 1.0
 */
template <size_t I, typename T, typename U>
struct std::tuple_element<I, museqa::utility::pair_t<T, U>>
  : std::tuple_element<I, museqa::utility::tuple_t<T, U>> {};

/**
 * Retrieves the deconstruction type of a n-tuple's element.
 * @tparam I The index of the requested tuple element.
 * @tparam T The n-tuple's elements' type.
 * @tparam N The total number of elements in the n-tuple.
 * @since 1.0
 */
template <size_t I, typename T, size_t N>
struct std::tuple_element<I, museqa::utility::ntuple_t<T, N>>
  : museqa::identity_t<museqa::utility::tuple_element_t<museqa::utility::ntuple_t<T, N>, I>> {};
