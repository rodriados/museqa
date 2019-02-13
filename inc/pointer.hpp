/**
 * Multiple Sequence Alignment pointer header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef POINTER_HPP_INCLUDED
#define POINTER_HPP_INCLUDED

#include <cstdint>
#include <cstddef>
#include <utility>

#include "utils.hpp"

/**
 * Type of function used for freeing pointers.
 * @tparam T The pointer type.
 * @since 0.1.1
 */
template <typename T>
using Deleter = Functor<void(Pointer<T>)>;

namespace pointer
{
    /**
     * The function for pointer deletion.
     * @tparam T The pointer type.
     * @param ptr The pointer to be deleted.
     */
    template <typename T>
    inline auto deleter(Pointer<T> ptr) -> typename std::enable_if<!std::is_array<T>::value, void>::type
    {
        delete ptr;
    }

    /**
     * The function for array pointer deletion.
     * @tparam T The pointer type.
     * @param ptr The array pointer to be deleted.
     */
    template <typename T>
    inline auto deleter(Pointer<T> ptr) -> typename std::enable_if<std::is_array<T>::value, void>::type
    {
        delete[] ptr;
    }
};

/**
 * Carries a raw pointer, possibly with its deleter function.
 * @tparam T The pointer type.
 * @since 0.1.1
 */
template <typename T>
struct RawPointer
{
    static_assert(!std::is_reference<T>::value, "Cannot create pointer to a reference.");
    static_assert(!std::is_function<T>::value, "Cannot create pointer to a function.");
    
    const Pointer<T> ptr = nullptr;                     /// The pointer itself.
    const Deleter<T> delfunc = pointer::deleter<T>;     /// The pointer deleter function.

    RawPointer() = default;
    RawPointer(const RawPointer<T>&) = default;
    RawPointer(RawPointer<T>&&) = default;

    /**
     * Initializes a new pointer storage object.
     * @param ptr The raw pointer to be held.
     * @param delfunc The deleter function for pointer.
     */
    inline RawPointer(Pointer<T> ptr, Deleter<T> delfunc = nullptr)
    :   ptr {ptr}
    ,   delfunc {delfunc ? delfunc : pointer::deleter<T>}
    {}

    RawPointer<T>& operator=(const RawPointer<T>&) = default;
    RawPointer<T>& operator=(RawPointer<T>&&) = default;

    /**
     * Converts to universal pointer type.
     * @return The pointer converted to universal type.
     */
    inline operator Pointer<T>() const
    {
        return ptr;
    }
};

namespace pointer
{
    /**
     * Counts the number of references of a given pointer.
     * @tparam T The pointer type.
     * @since 0.1.1
     */
    template <typename T>
    struct MetaPointer : public RawPointer<T>
    {
        size_t count = 0;               /// The number of existing references to pointer.

        /**
         * Initializes a new pointer counter.
         * @param ptr The raw pointer to be held.
         * @param delfunc The deleter function for pointer.
         */
        inline MetaPointer(Pointer<T> ptr, Deleter<T> delfunc)
        :   RawPointer<T> {ptr, delfunc}
        ,   count {1}
        {}

        /**
         * Deletes the raw pointer by calling its deleter function.
         * @see MetaPointer::MetaPointer
         */
        inline ~MetaPointer() noexcept
        {
            (this->delfunc)(this->ptr);
        }
    };

    /**
     * Creates a new pointer context from given arguments.
     * @tparam T The pointer type.
     * @param ptr The pointer to be put into context.
     * @param delfunc The delete functor.
     * @return New instance.
     */
    template <typename T>
    inline MetaPointer<T> *acquire(Pointer<T> ptr, Deleter<T> delfunc = nullptr)
    {
        return new MetaPointer<T> {ptr, delfunc};
    }

    /**
     * Acquires access to an already existing pointer.
     * @tparam T The pointer type.
     * @param ptr The pointer to be acquired.
     * @return The acquired pointer.
     */
    template <typename T>
    inline MetaPointer<T> *acquire(MetaPointer<T> *ptr)
    {
        ptr && ++ptr->count;
        return ptr;
    }

    /**
     * Releases access to pointer, and deletes it if needed.
     * @tparam T The pointer type.
     * @param ptr The pointer to be released.
     * @see pointer::acquire
     */
    template <typename T>
    inline void release(MetaPointer<T> *ptr)
    {
        if(ptr && --ptr->count <= 0)
            delete ptr;
    }
};

/**
 * The base of a smart pointer. This class can be used to represent a pointer that is
 * automatically deleted when all references to it have been destroyed.
 * @tparam T The type of pointer to be held.
 * @since 0.1.1
 */
template <typename T>
class BasePointer
{
    static_assert(!std::is_reference<T>::value, "Cannot create pointer to a reference.");
    static_assert(!std::is_function<T>::value, "Cannot create pointer to a function.");

    protected:
        pointer::MetaPointer<T> *meta = nullptr;    /// The pointer metadata.
        Pointer<T> ptr = nullptr;                   /// The encapsulated pointer.

    public:
        inline BasePointer() = default;

        /**
         * Builds a new instance from a raw pointer.
         * @param ptr The pointer to be encapsulated.
         * @param delfunc The delete functor.
         */
        inline BasePointer(Pointer<T> ptr, Deleter<T> delfunc = nullptr)
        :   meta {pointer::acquire<T>(ptr, delfunc)}
        ,   ptr {ptr}
        {}

        /**
         * Gets reference to an already existing pointer.
         * @param other The reference to be acquired.
         */
        inline BasePointer(const BasePointer<T>& other)
        :   meta {pointer::acquire(other.meta)}
        ,   ptr {other.ptr}
        {}

        /**
         * Acquires a moved reference to an already existing pointer.
         * @param other The reference to be moved.
         */
        inline BasePointer(BasePointer<T>&& other)
        :   meta {other.meta}
        ,   ptr {other.ptr}
        {
            other.reset();
        }

        /**
         * Builds a new instance from a raw pointer instance.
         * @tparam U The given raw pointer type.
         * @param raw The raw pointer object.
         */
        template <typename U = T>
        inline BasePointer(const RawPointer<U>& raw)
        :   BasePointer {raw.ptr, raw.delfunc}
        {}

        /**
         * Releases the acquired pointer reference.
         * @see BasePointer::BasePointer
         */
        inline ~BasePointer() noexcept
        {
            pointer::release(meta);
        }

        /**
         * The copy-assignment operator.
         * @param other The reference to be acquired.
         * @return This pointer object.
         */
        inline BasePointer<T>& operator=(const BasePointer<T>& other)
        {
            pointer::release(meta);

            meta = pointer::acquire(other.meta);
            ptr = other.ptr;

            return *this;
        }

        /**
         * The move-assignment operator.
         * @param other The reference to be acquired.
         * @return This pointer object.
         */
        inline BasePointer<T>& operator=(BasePointer<T>&& other) noexcept
        {
            pointer::release(meta);

            meta = other.meta;
            ptr = other.ptr;
            other.reset();

            return *this;
        }

        /**
         * Dereferences the pointer.
         * @return The pointed object.
         */
        __host__ __device__ inline Pure<T>& operator*() const
        {
            return *ptr;
        }

        /**
         * Gives access to the raw pointer.
         * @return The raw pointer.
         */
        __host__ __device__ inline Pointer<T> operator&() const
        {
            return ptr;
        }

        /**
         * Gives access to raw pointer using the dereference operator.
         * @return The raw pointer.
         */
        __host__ __device__ inline Pointer<T> operator->() const
        {
            return ptr;
        }

        /**
         * Checks if the stored pointer is not null.
         * @return Is the pointer not null?
         */
        __host__ __device__ inline operator bool() const
        {
            return ptr != nullptr;
        }

        /**
         * Gives access to raw pointer.
         * @return The raw pointer.
         */
        __host__ __device__ inline Pointer<T> get() const
        {
            return ptr;
        }

        /**
         * Gives access to the pointer deleter.
         * @return The pointer deleter.
         */
        inline Deleter<T> getDeleter() const
        {
            return meta ? meta->delfunc : nullptr;
        }

        /**
         * Informs the number of references created to pointer.
         * @return The number of references to pointer.
         */
        inline size_t useCount() const
        {
            return meta ? meta->count : 0;
        }

        /**
         * Resets the pointer manager to an empty state.
         * @see BasePointer::BasePointer
         */
        inline void reset()
        {
            meta = nullptr;
            ptr = nullptr;
        }
};

/**
 * Represents a smart pointer. This class can be used to represent a pointer that is
 * deleted automatically when all references to it have been destroyed.
 * @tparam T The type of pointer to be held.
 * @since 0.1.1
 */
template <typename T>
class AutoPointer : public BasePointer<T>
{
    public:
        inline AutoPointer() = default;
        inline AutoPointer(const AutoPointer<T>&) = default;
        inline AutoPointer(AutoPointer<T>&&) = default;

        using BasePointer<T>::BasePointer;

        AutoPointer<T>& operator=(const AutoPointer<T>&) = default;
        AutoPointer<T>& operator=(AutoPointer<T>&&) = default;

        /**
         * Converts to universal pointer type.
         * @return The pointer converted to universal type.
         */
        __host__ __device__ inline operator Pointer<T>() const
        {
            return this->ptr;
        }

        /**
         * Gives access to an object in an array pointer offset.
         * @param offset The offset to be accessed.
         * @return The requested object instance.
         */
        template <typename U = T>
        __host__ __device__ inline auto operator[](ptrdiff_t offset)
        -> typename std::enable_if<std::is_array<U>::value, Pure<T>&>::type
        {
            return getOffset(offset);
        }

        /**
         * Gives access to a const-qualified offset object instance.
         * @param offset The offset to pointer.
         * @return The offset object instance.
         */
        template <typename U = T>
        __host__ __device__ inline auto getOffset(ptrdiff_t offset) const
        -> typename std::enable_if<std::is_array<U>::value, Pure<T>&>::type
        {
            return this->ptr[offset];
        }
};

#endif