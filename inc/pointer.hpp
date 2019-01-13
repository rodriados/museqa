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

#include "cuda.cuh"
#include "utils.hpp"

/**
 * Type of function to use for freeing pointers.
 * @tparam T The pointer type.
 * @since 0.1.1
 */
template <typename T>
using Deleter = void (*)(Pure<T> *);

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

    /**
     * Holds raw pointer information.
     * @tparam T The pointer type.
     * @since 0.1.1
     */
    template <typename T>
    struct RawPointer
    {
        static_assert(!std::is_reference<T>::value, "Cannot create pointer to a reference.");

        Pure<T> * const ptr = nullptr;          /// The pointer itself
        const Deleter<T> delfunc = deleter<T>;  /// The pointer deleter function.
        size_t count = 0;                       /// The number of references to pointer.

        /**
         * Initializes a new raw pointer holder.
         * @param ptr The raw pointer to be held.
         * @param delfunc The deleter function for pointer.
         */
        inline RawPointer(Pure<T> * const ptr, const Deleter<T>& delfunc)
        :   ptr {ptr}
        ,   delfunc {delfunc}
        ,   count {1}
        {}

        /**
         * Deletes the raw pointer by calling its deleter function.
         * @see RawPointer::RawPointer
         */
        inline ~RawPointer() noexcept
        {
            (delfunc)(ptr);
        }

        /**
         * Creates a new pointer context from given arguments.
         * @param ptr The pointer to be put into context.
         * @param delfunc The delete functor.
         * @return New instance.
         */
        inline static RawPointer<T> *acquire(Pure<T> * const ptr, const Deleter<T>& delfunc = nullptr)
        {
            return new RawPointer<T> {ptr, delfunc ? delfunc : deleter<T>};
        }

        /**
         * Acquires access to an already existing pointer.
         * @param target The pointer to be acquired.
         * @return The acquired pointer.
         */
        inline static RawPointer<T> *acquire(RawPointer<T> *target)
        {
            target && ++target->count;
            return target;
        }

        /**
         * Releases access to pointer, and deletes it if needed.
         * @param target The pointer to be released.
         * @see RawPointer::acquire
         */
        inline static void release(RawPointer<T> *target)
        {
            if(target && --target->count <= 0)
                delete target;
        }
    };

    /**
     * Manages a raw pointer and all metadata directly related to it.
     * @tparam T The pointer type.
     * @since 0.1.1
     */
    template <typename T>
    class Manager
    {
        protected:
            RawPointer<T> *raw = nullptr;       /// The raw pointer holder.

        public:
            Manager() = default;

            /**
             * Builds a new instance from pointer to acquire.
             * @param ptr The pointer to acquire.
             * @param delfunc The deleter function.
             */
            inline Manager(Pure<T> * const ptr, const Deleter<T>& delfunc = nullptr)
            :   raw {RawPointer<T>::acquire(ptr, delfunc)}
            {}

            /**
             * Builds a new instance for an already existing pointer.
             * @param other The manager to acquire pointer from.
             */
            inline Manager(const Manager<T>& other)
            :   raw {RawPointer<T>::acquire(other.raw)}
            {}

            /**
             * Builds a new instance by moving an already existing pointer.
             * @param other The manager to acquire pointer from.
             */
            inline Manager(Manager<T>&& other)
            :   raw {other.raw}
            {
                other.raw = nullptr;
            }

            /**
             * Releases a reference to the pointer.
             * @see Manager::Manager
             */
            inline ~Manager() noexcept
            {
                RawPointer<T>::release(raw);
            }

            /**
             * The copy-assignment operator.
             * @param other The instance to be copied.
             * @return The current object.
             */
            inline Manager<T>& operator=(const Manager<T>& other)
            {
                RawPointer<T>::release(raw);

                raw = RawPointer<T>::acquire(other.raw);
                return *this;
            }

            /**
             * The move-assignment operator.
             * @param other The instance to be copied.
             * @return The current object.
             */
            inline Manager<T>& operator=(Manager<T>&& other)
            {
                RawPointer<T>::release(raw);

                raw = other.raw;
                other.raw = nullptr;
                return *this;
            }

            /**
             * Checks whether this reference is still valid.
             * @see Manager::Manager
             */
            inline operator bool() const
            {
                return raw
                    && raw->ptr;
            }

            /**
             * Gives access to raw pointer.
             * @return The raw pointer.
             */
            inline Pure<T> *get() const
            {
                return raw->ptr;
            }

            /**
             * Gives access to pointer deleter.
             * @return The pointer deleter.
             */
            inline const Deleter<T>& getDeleter() const
            {
                return raw->delfunc;
            }

            /**
             * Informs the number of references created to pointer.
             * @return The number of references to pointer.
             */
            inline const size_t useCount() const
            {
                return raw->count;
            }
    };

    /**
     * Represents a smart pointer base. This class can be used to represent a pointer that is
     * deleted automatically when all references to it have been destroyed.
     * @tparam T The type of pointer to be held.
     * @since 0.1.1
     */
    template <typename T>
    class BasePtr
    {
        protected:
            Pure<T> *ptr = nullptr;     /// The raw pointer.
            Manager<T> manager;         /// The pointer manager.

        public:
            BasePtr() = default;
            BasePtr(const BasePtr<T>&) = default;

            /**
             * Builds a new instance from a raw pointer.
             * @param ptr The pointer to be encapsulated.
             * @param delfunc The delete functor.
             */
            inline BasePtr(Pure<T> * const ptr, const Deleter<T>& delfunc = nullptr)
            :   ptr {ptr}
            ,   manager {ptr, delfunc}
            {}

            /**
             * The move constructor. Builds a copy of an instance, by moving.
             * @param other The instance to be moved.
             */
            inline BasePtr(BasePtr<T>&& other)
            :   ptr {other.ptr}
            ,   manager {std::move(other.manager)}
            {
                other.reset();
            }

            /**
             * The move-assignment operator.
             * @param other The instance to be moved.
             * @return The current object.
             */
            inline BasePtr<T>& operator=(BasePtr<T>&& other) noexcept
            {
                ptr = other.ptr;
                manager = std::move(other.manager);
                other.reset();
            }

            BasePtr<T>& operator=(const BasePtr<T>&) noexcept = default;

            /**
             * Dereferences the pointer.
             * @return The pointed object.
             */
            cudadecl inline Pure<T>& operator*()
            {
                return *ptr;
            }

            /**
             * Gives access to the raw pointer.
             * @return The raw pointer.
             */
            cudadecl inline Pure<T> *operator&()
            {
                return ptr;
            }

            /**
             * Gives access to raw pointer for dereference operator.
             * @return The raw pointer.
             */
            cudadecl inline Pure<T> *operator->()
            {
                return ptr;
            }

            /**
             * Checks if the stored pointer is not null.
             * @return Is the pointer not null?
             */
            cudadecl inline operator bool() const
            {
                return (bool) ptr;
            }

            /**
             * Gives access to const-qualified raw pointer.
             * @return The raw pointer.
             */
            cudadecl inline Pure<T> *get() const
            {
                return ptr;
            }

            /**
             * Gives access to pointer deleter.
             * @return The pointer deleter.
             */
            inline const Deleter<T>& getDeleter() const
            {
                return manager.getDeleter();
            }

            /**
             * Resets the pointer manager to an empty state.
             * @see BasePtr::BasePtr
             */
            inline void reset()
            {
                ptr = nullptr;
                manager = Manager<T> {};
            }

            /**
             * Informs the number of references created to pointer.
             * @return The number of references to pointer.
             */
            inline size_t useCount() const
            {
                return manager.useCount();
            }
    };
};

/**
 * Aliasing the base pointer object.
 * @tparam T The pointer type.
 * @since 0.1.1
 */
template <typename T>
using BasePtr = pointer::BasePtr<T>;

/**
 * Represents a smart pointer. This class can be used to represent a pointer that is
 * deleted automatically when all references to it have been destroyed.
 * @tparam T The type of pointer to be held.
 * @since 0.1.1
 */
template <typename T, typename E = void>
class SmartPtr : public BasePtr<T>
{
    public:
        inline SmartPtr() = default;
        inline SmartPtr(const SmartPtr&) = default;
        inline SmartPtr(SmartPtr&&) = default;

        using BasePtr<T>::BasePtr;

        SmartPtr<T, E>& operator=(const SmartPtr<T, E>&) = default;
        SmartPtr<T, E>& operator=(SmartPtr<T, E>&&) = default;
};

/**
 * Represents a smart array pointer. This class can be used to represent a pointer that
 * is deleted automatically when all references to it have been destroyed.
 * @tparam T The type of pointer to be held.
 * @since 0.1.1
 */
template <typename T>
class SmartPtr<T, typename std::enable_if<std::is_array<T>::value>::type> : public BasePtr<T>
{
    public:
        inline SmartPtr() = default;
        using BasePtr<T>::BasePtr;

        /**
         * Gives access to an object in a pointer offset.
         * @param offset The offset to be accessed.
         * @return The requested object instance.
         */
        cudadecl inline Pure<T>& operator[](ptrdiff_t offset)
        {
            return this->ptr[offset];
        }

        /**
         * Gives access to a const-qualified offset object instance.
         * @param offset The offset to pointer.
         * @return The offset object instance.
         */
        cudadecl inline Pure<T>& getOffset(ptrdiff_t offset) const
        {
            return this->ptr[offset];
        }
};

#endif
