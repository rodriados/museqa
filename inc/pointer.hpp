/**
 * Multiple Sequence Alignment pointer header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef POINTER_HPP_INCLUDED
#define POINTER_HPP_INCLUDED

#pragma once

#include <cstdint>
#include <cstddef>
#include <utility>

#include "device.cuh"

/**
 * Purifies the type by removing array extent.
 * @tparam T The type to be cleaned.
 * @since 0.1.alpha
 */
template<typename T>
using Pure = typename std::remove_extent<T>::type;

/**
 * Type of function to use for freeing pointers.
 * @tparam T The pointer type.
 * @since 0.1.alpha
 */
template<typename T>
using Deleter = void (*)(Pure<T> *);

namespace pointer
{
    /**
     * The function for pointer deletion.
     * @tparam T The pointer type.
     * @param ptr The pointer to be deleted.
     */
    template<typename T, typename std::enable_if<!std::is_array<T>::value, int>::type = 0>
    inline void deleter(Pure<T> *ptr)
    {
        delete ptr;
    }

    /**
     * The function for array pointer deletion.
     * @tparam T The pointer type.
     * @param ptr The array pointer to be deleted.
     */
    template<typename T, typename std::enable_if<std::is_array<T>::value, int>::type = 0>
    inline void deleter(Pure<T> *ptr)
    {
        delete[] ptr;
    }

    /**
     * Holds a pure type pointer and keeps its reference counter.
     * @tparam T The pointer type.
     * @since 0.1.alpha
     */
    template<typename T>
    class Holder
    {
        protected:
            typedef Pure<T> PureT;

        protected:
            Deleter<T> dfunc = deleter<T>;  /// The pointer delete functor.
            PureT *ptr = nullptr;           /// The raw pointer.
            size_t count = 0;              /// The pointer references counter.

        public:
            /**
             * Gives access to raw pointer.
             * @return The raw pointer.
             */
            cudadecl inline PureT *getRaw()
            {
                return this->ptr;
            }

            /**
             * Creates a new pointer holder from given arguments.
             * @param ptr The pointer to be acquired.
             * @param dfunc The delete functor.
             * @return New holder instance.
             */
            static inline Holder<T> *acquire(PureT *ptr, const Deleter<T>& dfunc = nullptr)
            {
                return new Holder<T>(ptr, dfunc);
            }

            /**
             * Acquires access to an already existing pointer holder.
             * @param target The holder to be acquired.
             * @return The acquired holder.
             */
            static inline Holder<T> *acquire(Holder<T> *target)
            {
                target && ++(target->count);
                return target;
            }

            /**
             * Releases access to pointer holder, and deletes it if needed.
             * @param target The holder to be released.
             */
            static inline void release(Holder<T> *target)
            {
                if(target && --(target->count) <= 0)
                    delete target;
            }

        protected:
            /**
             * Builds a new instance from pointer to acquire.
             * @param ptr The pointer to acquire.
             * @param dfunc The delete function.
             */
            inline Holder(PureT *ptr, const Deleter<T>& dfunc = nullptr)
            :   dfunc(dfunc ? dfunc : deleter<T>)
            ,   ptr(ptr)
            ,   count(1) {}

            /**
             * Deletes the pointer. This constructor shall only be called
             * when there are no more pointer references left.
             */
            inline ~Holder() noexcept
            {
                (this->dfunc)(this->ptr);
            }
    };
};

/**
 * Represents the base class of a smart pointer.
 * @tparam T The type of pointer to be held.
 * @since 0.1.alpha
 */
template<typename T>
class BasePointer
{
    protected:
        typedef Pure<T> PureT;

    protected:
        pointer::Holder<T> *meta = nullptr; /// The pointer metadata.
        PureT *ptr = nullptr;               /// The raw pointer.

    public:
        BasePointer() = default;

        /**
         * The copy constructor. Builds a copy of an instance.
         * @param other The instance to be copied.
         */
        inline BasePointer(const BasePointer<T>& other)
        :   meta(pointer::Holder<T>::acquire(other.meta))
        ,   ptr(other.ptr) {}

        /**
         * The move constructor. Builds a copy of an instance, by moving.
         * @param other The instance to be moved.
         */
        inline BasePointer(BasePointer<T>&& other)
        :   meta(other.meta)
        ,   ptr(other.ptr)
        {
            other.meta = nullptr;
            other.ptr = nullptr;
        }

        /**
         * Builds a new instance from a raw pointer.
         * @param ptr The pointer to be encapsulated.
         * @param dfunc The delete functor.
         */
        inline BasePointer(PureT *ptr, const Deleter<T>& dfunc = nullptr)
        :   meta(pointer::Holder<T>::acquire(ptr, dfunc))
        ,   ptr(ptr) {}

        /**
         * Destroys a pointer instance. The raw pointer will only be freed if the
         * instance counter drops to zero or less.
         */
        inline ~BasePointer() noexcept
        {
            pointer::Holder<T>::release(this->meta);
        }

        /**
         * The copy-assignment operator.
         * @param other The instance to be copied.
         * @return The current object.
         */
        inline BasePointer<T>& operator=(const BasePointer<T>& other)
        {
            if(this->meta != other.meta) {
                pointer::Holder<T>::release(this->meta);
                this->meta = pointer::Holder<T>::acquire(other.meta);
                this->ptr = other.ptr;
            }

            return *this;
        }

        /**
         * The move-assignment operator.
         * @param other The instance to be moved.
         * @return The current object.
         */
        inline BasePointer<T>& operator=(BasePointer<T>&& other) noexcept
        {
            if(this->meta != other.meta) {
                pointer::Holder<T>::release(this->meta);
                this->meta = other.meta;
                this->ptr = other.ptr;
                other.meta = nullptr;
                other.ptr = nullptr;
            }

            return *this;
        }

        /**
         * The assignment operator. Changes the pointer.
         * @param ptr The new pointer target.
         * @return The current object.
         */
        inline BasePointer<T>& operator=(PureT *ptr)
        {
            pointer::Holder<T>::release(this->meta);
            this->meta = pointer::Holder<T>::acquire(ptr);
            this->ptr = ptr;
            return *this;
        }

        /**
         * Dereferences the pointer.
         * @return The pointed object.
         */
        cudadecl inline PureT& operator*()
        {
            return *this->ptr;
        }

        /**
         * Gives access to the raw pointer.
         * @return The raw pointer.
         */
        cudadecl inline PureT *operator&()
        {
            return this->ptr;
        }

        /**
         * Gives access to raw pointer for dereference operator.
         * @return The raw pointer.
         */
        cudadecl inline PureT *operator->()
        {
            return this->ptr;
        }

        /**
         * Gives access to const-qualified raw pointer.
         * @return The raw pointer.
         */
        cudadecl inline PureT *getRaw() const
        {
            return this->ptr;
        }
};

/**
 * Represents a smart pointer. This class can be used to represent a pointer that is
 * deleted automatically when all references to it have been destroyed.
 * @tparam T The type of pointer to be held.
 * @since 0.1.alpha
 */
template<typename T, typename E = void>
class SharedPointer : public BasePointer<T>
{
    protected:
        typedef SharedPointer<T, E> this_t;

    public:
        SharedPointer() = default;
        
        using BasePointer<T>::BasePointer;

        this_t& operator=(const this_t&) = default;
        this_t& operator=(this_t&&) = default;
};

/**
 * Represents a smart array pointer. This class can be used to represent a pointer that
 * is deleted automatically when all references to it have been destroyed.
 * @tparam T The type of pointer to be held.
 * @tparam D The pointer deleter function.
 * @since 0.1.alpha
 */
template<typename T>
class SharedPointer<T, typename std::enable_if<std::is_array<T>::value>::type> : public BasePointer<T>
{
    protected:
        typedef Pure<T> PureT;
        typedef SharedPointer<T, typename std::enable_if<std::is_array<T>::value>::type> this_t;

    public:
        SharedPointer() = default;

        using BasePointer<T>::BasePointer;

        /**
         * Creates a new instance as an offset to a pointer.
         * @param other The base pointer instance.
         * @param displ The new pointer displacement.
         */
        inline SharedPointer(const this_t& other, ptrdiff_t displ = 0)
        :   BasePointer<T>(other)
        {
            this->ptr = other.ptr + displ;
        }

        this_t& operator=(const this_t&) = default;
        this_t& operator=(this_t&&) = default;

        /**
         * Gives access to an object in a pointer offset.
         * @param offset The offset to be accessed.
         * @return The requested object instance.
         */
        cudadecl inline PureT& operator[](ptrdiff_t offset)
        {
            return this->ptr[offset];
        }

        /**
         * Creates a new offset pointer instance.
         * @param offset The offset to pointer.
         * @return New offset pointer instane.
         */
        inline this_t operator+(ptrdiff_t offset) const
        {
            return this_t(*this, offset);
        }

        /**
         * Gives access to a const-qualified offset object instance.
         * @param offset The offset to pointer.
         * @return The offset object instance.
         */
        cudadecl inline PureT& getOffset(ptrdiff_t offset) const
        {
            return this->ptr[offset];
        }
};

#endif
