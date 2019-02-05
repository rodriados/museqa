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
 * A universal pointer type. All pointer objects should be convertible to it.
 * @tparam T The pointer type.
 * @since 0.1.1
 */
template <typename T>
using Pointer = Pure<T> *;

/**
 * Type of function to use for freeing pointers.
 * @tparam T The pointer type.
 * @since 0.1.1
 */
template <typename T>
using Deleter = void (*)(Pointer<T>);

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

    /**
     * Holds raw pointer information.
     * @tparam T The pointer type.
     * @since 0.1.1
     */
    template <typename T>
    struct Storage
    {
        static_assert(!std::is_reference<T>::value, "Cannot create pointer to a reference.");
        
        const Pointer<T> ptr = nullptr;             /// The pointer itself.
        const Deleter<T> delfunc = deleter<T>;      /// The pointer deleter function.

        Storage() = default;
        Storage(const Storage<T>&) = default;
        Storage(Storage<T>&&) = default;

        /**
         * Initializes a new pointer storage object.
         * @param ptr The raw pointer to be held.
         * @param delfunc The deleter function for pointer.
         */
        inline Storage(const Pointer<T> ptr, const Deleter<T> delfunc = nullptr)
        :   ptr {ptr}
        ,   delfunc {delfunc ? delfunc : deleter<T>}
        {}

        Storage<T>& operator=(const Storage<T>&) = default;
        Storage<T>& operator=(Storage<T>&&) = default;

        /**
         * Converts to universal pointer type.
         * @return The pointer converted to universal type.
         */
        inline operator Pointer<T>() const
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
    struct Observer : public Storage<T>
    {
        size_t count = 0;                       /// The number of references to pointer.

        /**
         * Initializes a new pointer counter.
         * @param ptr The raw pointer to be held.
         * @param delfunc The deleter function for pointer.
         */
        inline Observer(const Pointer<T> ptr, const Deleter<T> delfunc)
        :   Storage<T> {ptr, delfunc}
        ,   count {1}
        {}

        /**
         * Deletes the raw pointer by calling its deleter function.
         * @see Observer::Observer
         */
        inline ~Observer() noexcept
        {
            (this->delfunc)(this->ptr);
        }

        /**
         * Creates a new pointer context from given arguments.
         * @param ptr The pointer to be put into context.
         * @param delfunc The delete functor.
         * @return New instance.
         */
        inline static Observer<T> *acquire(const Pointer<T> ptr, const Deleter<T> delfunc = nullptr)
        {
            return new Observer<T> {ptr, delfunc};
        }

        /**
         * Acquires access to an already existing pointer.
         * @param target The pointer to be acquired.
         * @return The acquired pointer.
         */
        inline static Observer<T> *acquire(Observer<T> *target)
        {
            target && ++target->count;
            return target;
        }

        /**
         * Releases access to pointer, and deletes it if needed.
         * @param target The pointer to be released.
         * @see Observer::acquire
         */
        inline static void release(Observer<T> *target)
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
            Observer<T> *raw = nullptr;   /// The raw info storage.

        public:
            Manager() = default;

            /**
             * Builds a new instance from pointer to acquire.
             * @param ptr The pointer to acquire.
             * @param delfunc The deleter function.
             */
            inline Manager(const Pointer<T> ptr, const Deleter<T> delfunc = nullptr)
            :   raw {Observer<T>::acquire(ptr, delfunc)}
            {}

            /**
             * Builds a new instance for an already existing pointer.
             * @param other The manager to acquire pointer from.
             */
            inline Manager(const Manager<T>& other)
            :   raw {Observer<T>::acquire(other.raw)}
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
                Observer<T>::release(raw);
            }

            /**
             * The copy-assignment operator.
             * @param other The instance to be copied.
             * @return The current object.
             */
            inline Manager<T>& operator=(const Manager<T>& other)
            {
                Observer<T>::release(raw);

                raw = Observer<T>::acquire(other.raw);
                return *this;
            }

            /**
             * The move-assignment operator.
             * @param other The instance to be copied.
             * @return The current object.
             */
            inline Manager<T>& operator=(Manager<T>&& other)
            {
                Observer<T>::release(raw);

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
            inline const Pointer<T> get() const
            {
                return raw->ptr;
            }

            /**
             * Gives access to pointer deleter.
             * @return The pointer deleter.
             */
            inline const Deleter<T> getDeleter() const
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
            Pointer<T> ptr = nullptr;       /// The raw pointer.
            Manager<T> manager;             /// The pointer manager.

        public:
            BasePointer() = default;
            BasePointer(const BasePointer<T>&) = default;

            /**
             * Builds a new instance from a raw pointer.
             * @param ptr The pointer to be encapsulated.
             * @param delfunc The delete functor.
             */
            inline BasePointer(const Pointer<T> ptr, const Deleter<T> delfunc = nullptr)
            :   ptr {ptr}
            ,   manager {ptr, delfunc}
            {}

            /**
             * Builds a new instance from a pointer storage instance.
             * @param storage The pointer storage object.
             */
            template <typename U = T>
            inline BasePointer(const Storage<U>& storage)
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
            cudadecl inline Pure<T>& operator*() const
            {
                return *ptr;
            }

            /**
             * Gives access to the raw pointer.
             * @return The raw pointer.
             */
            cudadecl inline const Pointer<T> operator&() const
            {
                return ptr;
            }

            /**
             * Gives access to raw pointer for dereference operator.
             * @return The raw pointer.
             */
            cudadecl inline const Pointer<T> operator->() const
            {
                return ptr;
            }

            /**
             * Checks if the stored pointer is not null.
             * @return Is the pointer not null?
             */
            cudadecl inline operator bool() const
            {
                return ptr != nullptr;
            }

            /**
             * Gives access to const-qualified raw pointer.
             * @return The raw pointer.
             */
            cudadecl inline const Pointer<T> get() const
            {
                return ptr;
            }

            /**
             * Gives access to pointer deleter.
             * @return The pointer deleter.
             */
            inline const Deleter<T> getDeleter() const
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
using RawPointer = pointer::Storage<T>;

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
        cudadecl inline operator Pointer<T>() const
        {
            return this->ptr;
        }

        /**
         * Gives access to an object in a pointer offset.
         * @param offset The offset to be accessed.
         * @return The requested object instance.
         */
        cudadecl inline Pure<T>& operator[](ptrdiff_t offset)
        {
            static_assert(std::is_array<T>::value, "Pointer type is not an array");
            return this->ptr[offset];
        }

        /**
         * Gives access to a const-qualified offset object instance.
         * @param offset The offset to pointer.
         * @return The offset object instance.
         */
        template <typename U = T>
        cudadecl inline Pure<T>& getOffset(ptrdiff_t offset) const
        {
            static_assert(std::is_array<T>::value, "Pointer type is not an array");
            return this->ptr[offset];
        }
};

#endif
