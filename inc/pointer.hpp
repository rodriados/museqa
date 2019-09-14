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
using Deleter = Functor<void(Pure<T> *)>;

namespace pointer
{
    /**
     * The function for pointer deletion.
     * @tparam T The pointer type.
     * @param ptr The pointer to be deleted.
     */
    template <typename T>
    inline auto deleter(Pure<T> *ptr) -> typename std::enable_if<!std::is_array<T>::value, void>::type
    {
        delete ptr;
    }

    /**
     * The function for array pointer deletion.
     * @tparam T The pointer type.
     * @param ptr The array pointer to be deleted.
     */
    template <typename T>
    inline auto deleter(Pure<T> *ptr) -> typename std::enable_if<std::is_array<T>::value, void>::type
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
    
    Pure<T> *ptr = nullptr;                         /// The pointer itself.
    Deleter<T> delfunc = pointer::deleter<T>;       /// The pointer deleter function.

    inline constexpr RawPointer() noexcept = default;
    inline constexpr RawPointer(const RawPointer&) noexcept = default;
    inline constexpr RawPointer(RawPointer&&) noexcept = default;

    /**
     * Initializes a new pointer storage object.
     * @param ptr The raw pointer to be held.
     * @param delfunc The deleter function for pointer.
     */
    inline constexpr RawPointer(Pure<T> *ptr, Deleter<T> delfunc = nullptr) noexcept
    :   ptr {ptr}
    ,   delfunc {!delfunc.isEmpty() ? delfunc : pointer::deleter<T>}
    {}

    inline RawPointer& operator=(const RawPointer&) noexcept = default;
    inline RawPointer& operator=(RawPointer&&) noexcept = default;

    /**
     * Converts to universal pointer type.
     * @return The pointer converted to universal type.
     */
    inline constexpr operator Pure<T> *() const noexcept
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
        inline MetaPointer(Pure<T> *ptr, Deleter<T> delfunc) noexcept
        :   RawPointer<T> {ptr, delfunc}
        ,   count {1}
        {}

        /**
         * Deletes the raw pointer by calling its deleter function.
         * @see MetaPointer::MetaPointer
         */
        inline ~MetaPointer()
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
    inline MetaPointer<T> *acquire(Pure<T> *ptr, Deleter<T> delfunc = nullptr) noexcept
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
    __host__ __device__ inline MetaPointer<T> *acquire(MetaPointer<T> *ptr) noexcept
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
    __host__ __device__ inline void release(MetaPointer<T> *ptr)
    {
#ifndef msa_compile_cuda
        if(ptr && --ptr->count <= 0)
            delete ptr;
#endif
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
        Pure<T> *ptr = nullptr;                     /// The encapsulated pointer.

    public:
        __host__ __device__ inline BasePointer() noexcept = default;

#ifndef msa_compile_cuda
        /**
         * Builds a new instance from a raw pointer.
         * @param ptr The pointer to be encapsulated.
         * @param delfunc The delete functor.
         */
        inline BasePointer(Pure<T> *ptr, Deleter<T> delfunc = nullptr) noexcept
        :   meta {pointer::acquire<T>(ptr, delfunc)}
        ,   ptr {ptr}
        {}
#else
        /**
         * Builds a new instance from a device pointer. This pointer is not, in
         * any manner, owned by the instance. This is just so that a device pointer
         * can successfully populate an object property securely.
         * @param ptr The device pointer to be encapsulated.
         */
        __device__ inline BasePointer(Pure<T> *ptr, Deleter<T> = nullptr) noexcept
        :   ptr {ptr}
        {}
#endif

        /**
         * Gets reference to an already existing pointer.
         * @param other The reference to be acquired.
         */
        __host__ __device__ inline BasePointer(const BasePointer& other) noexcept
        :   meta {pointer::acquire(other.meta)}
        ,   ptr {other.ptr}
        {}

        /**
         * Acquires a moved reference to an already existing pointer.
         * @param other The reference to be moved.
         */
        __host__ __device__ inline BasePointer(BasePointer&& other) noexcept
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
        inline BasePointer(const RawPointer<U>& raw) noexcept
        :   BasePointer {raw.ptr, raw.delfunc}
        {}

        /**
         * Releases the acquired pointer reference.
         * @see BasePointer::BasePointer
         */
        __host__ __device__ inline ~BasePointer()
        {
            pointer::release(meta);
        }

        /**
         * The copy-assignment operator.
         * @param other The reference to be acquired.
         * @return This pointer object.
         */
        __host__ __device__ inline BasePointer& operator=(const BasePointer& other)
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
        __host__ __device__ inline BasePointer& operator=(BasePointer&& other)
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
        __host__ __device__ inline Pure<T>& operator*() noexcept
        {
            return *ptr;
        }

        /**
         * Dereferences the constant pointer.
         * @return The constant pointed object.
         */
        __host__ __device__ inline const Pure<T>& operator*() const noexcept
        {
            return *ptr;
        }

        /**
         * Gives access to the raw pointer.
         * @return The raw pointer.
         */
        __host__ __device__ inline Pure<T> *operator&() noexcept
        {
            return ptr;
        }

        /**
         * Gives access to the raw constant pointer.
         * @return The raw constant pointer.
         */
        __host__ __device__ inline const Pure<T> *operator&() const noexcept
        {
            return ptr;
        }

        /**
         * Gives access to raw pointer using the dereference operator.
         * @return The raw pointer.
         */
        __host__ __device__ inline Pure<T> *operator->() noexcept
        {
            return ptr;
        }

        /**
         * Gives access to raw constant pointer using the dereference operator.
         * @return The raw constant pointer.
         */
        __host__ __device__ inline const Pure<T> *operator->() const noexcept
        {
            return ptr;
        }

        /**
         * Checks if the stored pointer is not null.
         * @return Is the pointer not null?
         */
        __host__ __device__ inline operator bool() const noexcept
        {
            return ptr != nullptr;
        }

        /**
         * Gives access to raw pointer.
         * @return The raw pointer.
         */
        __host__ __device__ inline const Pure<T> *get() const noexcept
        {
            return ptr;
        }

        /**
         * Gives access to the pointer deleter.
         * @return The pointer deleter.
         */
        inline const Deleter<T> getDeleter() const noexcept
        {
            return meta ? meta->delfunc : nullptr;
        }

        /**
         * Informs the number of references created to pointer.
         * @return The number of references to pointer.
         */
        inline size_t useCount() const noexcept
        {
            return meta ? meta->count : 0;
        }

        /**
         * Resets the pointer manager to an empty state.
         * @see BasePointer::BasePointer
         */
        __host__ __device__ inline void reset() noexcept
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
class Pointer : public BasePointer<T>
{
    public:
        __host__ __device__ inline Pointer() noexcept = default;
        __host__ __device__ inline Pointer(const Pointer&) noexcept = default;
        __host__ __device__ inline Pointer(Pointer&&) noexcept = default;

        using BasePointer<T>::BasePointer;

        __host__ __device__ inline Pointer& operator=(const Pointer&) = default;
        __host__ __device__ inline Pointer& operator=(Pointer&&) = default;

        /**
         * Converts to universal pointer type.
         * @return The pointer converted to universal type.
         */
        __host__ __device__ inline operator Pure<T> *() noexcept
        {
            return this->ptr;
        }

        /**
         * Converts to universal constant pointer type.
         * @return The constant pointer converted to universal type.
         */
        __host__ __device__ inline operator const Pure<T> *() const noexcept
        {
            return this->ptr;
        }

        /**
         * Gives access to an object in an array pointer offset.
         * @param offset The offset to be accessed.
         * @return The requested object instance.
         */
        __host__ __device__ inline Pure<T>& operator[](ptrdiff_t offset) noexcept
        {
            static_assert(!std::is_same<Pure<T>, T>::value, "only array pointers have offsets");
            return this->ptr[offset];
        }

        /**
         * Gives access to an constant object in an array pointer offset.
         * @param offset The offset to be accessed.
         * @return The requested constant object instance.
         */
        __host__ __device__ inline const Pure<T>& operator[](ptrdiff_t offset) const noexcept
        {
            static_assert(!std::is_same<Pure<T>, T>::value, "only array pointers have offsets");
            return this->ptr[offset];
        }

        /**
         * Gives access to an offset constant object instance.
         * @param offset The offset to pointer.
         * @return The offset object instance.
         */
        template <typename U = T>
        __host__ __device__ inline auto getOffset(ptrdiff_t offset) const noexcept
        -> typename std::enable_if<!std::is_same<Pure<U>, U>::value, const Pure<T>&>::type
        {
            return this->ptr[offset];
        }

        /**
         * Gets an instance to an offset pointer.
         * @param offset The requested offset.
         * @return The new offset pointer instance.
         */
        template <typename U = T>
        __host__ __device__ inline auto getOffsetPointer(ptrdiff_t offset) noexcept
        -> typename std::enable_if<!std::is_same<Pure<U>, U>::value, Pointer>::type
        {
            Pointer instance {*this};
            instance.ptr += offset;
            return instance;
        }
};

#endif