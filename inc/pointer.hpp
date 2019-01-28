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
    struct PointerStorage
    {
        static_assert(!std::is_reference<T>::value, "Cannot create pointer to a reference.");
        
        Pure<T> * const ptr = nullptr;          /// The pointer itself.
        const Deleter<T> delfunc = deleter<T>;  /// The pointer deleter function.

        /**
         * Initializes a new pointer storage object.
         * @param ptr The raw pointer to be held.
         * @param delfunc The deleter function for pointer.
         */
        inline PointerStorage(Pure<T> * const ptr, const Deleter<T>& delfunc)
        :   ptr {ptr}
        ,   delfunc {delfunc}
        {}

        /**
         * Allows easy conversion to built-in pointer type.
         * @return The built-in pointer.
         */
        inline operator Pure<T> *()
        {
            return ptr;
        }
    };

    /**
     * Counts the number of references of a given pointer.
     * @tparam T The pointer type.
     * @since 0.1.1
     */
    template <typename T>
    struct PointerKeeper : public PointerStorage<T>
    {
        size_t count = 0;                       /// The number of references to pointer.

        /**
         * Initializes a new pointer counter.
         * @param ptr The raw pointer to be held.
         * @param delfunc The deleter function for pointer.
         */
        inline PointerKeeper(Pure<T> * const ptr, const Deleter<T>& delfunc)
        :   PointerStorage<T> {ptr, delfunc}
        ,   count {1}
        {}

        /**
         * Deletes the raw pointer by calling its deleter function.
         * @see PointerKeeper::PointerKeeper
         */
        inline ~PointerKeeper() noexcept
        {
            (this->delfunc)(this->ptr);
        }

        /**
         * Creates a new pointer context from given arguments.
         * @param ptr The pointer to be put into context.
         * @param delfunc The delete functor.
         * @return New instance.
         */
        inline static PointerKeeper<T> *acquire(Pure<T> * const ptr, const Deleter<T>& delfunc = nullptr)
        {
            return new PointerKeeper<T> {ptr, delfunc ? delfunc : deleter<T>};
        }

        /**
         * Acquires access to an already existing pointer.
         * @param target The pointer to be acquired.
         * @return The acquired pointer.
         */
        inline static PointerKeeper<T> *acquire(PointerKeeper<T> *target)
        {
            target && ++target->count;
            return target;
        }

        /**
         * Releases access to pointer, and deletes it if needed.
         * @param target The pointer to be released.
         * @see PointerKeeper::acquire
         */
        inline static void release(PointerKeeper<T> *target)
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
            PointerKeeper<T> *raw = nullptr;   /// The raw info storage.

        public:
            Manager() = default;

            /**
             * Builds a new instance from pointer to acquire.
             * @param ptr The pointer to acquire.
             * @param delfunc The deleter function.
             */
            inline Manager(Pure<T> * const ptr, const Deleter<T>& delfunc = nullptr)
            :   raw {PointerKeeper<T>::acquire(ptr, delfunc)}
            {}

            /**
             * Builds a new instance for an already existing pointer.
             * @param other The manager to acquire pointer from.
             */
            inline Manager(const Manager<T>& other)
            :   raw {PointerKeeper<T>::acquire(other.raw)}
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
                PointerKeeper<T>::release(raw);
            }

            /**
             * The copy-assignment operator.
             * @param other The instance to be copied.
             * @return The current object.
             */
            inline Manager<T>& operator=(const Manager<T>& other)
            {
                PointerKeeper<T>::release(raw);

                raw = PointerKeeper<T>::acquire(other.raw);
                return *this;
            }

            /**
             * The move-assignment operator.
             * @param other The instance to be copied.
             * @return The current object.
             */
            inline Manager<T>& operator=(Manager<T>&& other)
            {
                PointerKeeper<T>::release(raw);

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
    class BasePointer
    {
        protected:
            Pure<T> *ptr = nullptr;     /// The raw pointer.
            Manager<T> manager;         /// The pointer manager.

        public:
            BasePointer() = default;
            BasePointer(const BasePointer<T>&) = default;

            /**
             * Builds a new instance from a raw pointer.
             * @param ptr The pointer to be encapsulated.
             * @param delfunc The delete functor.
             */
            inline BasePointer(Pure<T> * const ptr, const Deleter<T>& delfunc = nullptr)
            :   ptr {ptr}
            ,   manager {ptr, delfunc}
            {}

            /**
             * Builds a new instance from a pointer storage instance.
             * @param storage The pointer storage object.
             */
            inline BasePointer(const PointerStorage<T>& storage)
            :   BasePointer {storage.ptr, storage.delfunc}
            {}

            /**
             * The move constructor. Builds a copy of an instance, by moving.
             * @param other The instance to be moved.
             */
            inline BasePointer(BasePointer<T>&& other)
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
            inline BasePointer<T>& operator=(BasePointer<T>&& other) noexcept
            {
                ptr = other.ptr;
                manager = std::move(other.manager);
                other.reset();
            }

            BasePointer<T>& operator=(const BasePointer<T>&) = default;

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
             * @see BasePointer::BasePointer
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
using BasePointer = pointer::BasePointer<T>;

/**
 * Aliasing the raw pointer object type.
 * @tparam T The pointer type.
 * @since 0.1.1
 */
template <typename T>
using RawPointer = pointer::PointerStorage<T>;

/**
 * Represents a smart pointer. This class can be used to represent a pointer that is
 * deleted automatically when all references to it have been destroyed.
 * @tparam T The type of pointer to be held.
 * @since 0.1.1
 */
template <typename T, typename E = void>
class Pointer : public BasePointer<T>
{
    public:
        inline Pointer() = default;
        inline Pointer(const Pointer&) = default;
        inline Pointer(Pointer&&) = default;

        using BasePointer<T>::BasePointer;

        Pointer<T, E>& operator=(const Pointer<T, E>&) = default;
        Pointer<T, E>& operator=(Pointer<T, E>&&) = default;
};

/**
 * Represents a smart array pointer. This class can be used to represent a pointer that
 * is deleted automatically when all references to it have been destroyed.
 * @tparam T The type of pointer to be held.
 * @since 0.1.1
 */
template <typename T>
class Pointer<T, typename std::enable_if<std::is_array<T>::value>::type> : public BasePointer<T>
{
    public:
        inline Pointer() = default;
        using BasePointer<T>::BasePointer;

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
