/** 
 * Multiple Sequence Alignment sequence header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef SEQUENCE_HPP_INCLUDED
#define SEQUENCE_HPP_INCLUDED

#include <cstdint>
#include <ostream>
#include <string>

#include "utils.hpp"
#include "buffer.hpp"
#include "encoder.hpp"

/**
 * Holds an enconded sequence. The encoding pattern will used throughout all
 * steps: it saves up to a third of the required space and is easily revertable.
 * @since 0.1.1
 */
class Sequence : public Buffer<encoder::EncodedBlock>
{
    public:
        Sequence() = default;
        Sequence(const Sequence&) = default;
        Sequence(Sequence&&) = default;
        
        using Buffer<encoder::EncodedBlock>::Buffer;

        /**
         * Initializes a new compressed sequence.
         * @param ptr The pointer to buffer to be encoded.
         * @param size The buffer's size.
         */
        inline Sequence(Pointer<const char> ptr, size_t size)
        :   Buffer<encoder::EncodedBlock> {encoder::encode(ptr, size)}
        {}

        /**
         * Instantiates a new sequence.
         * @param string The string containing this sequence's data.
         */
        inline Sequence(const std::string& string)
        :   Sequence {string.data(), string.size()}
        {}

        Sequence& operator=(const Sequence&) = default;
        Sequence& operator=(Sequence&&) = default;

        /**
         * Retrieves the element at given offset.
         * @param offset The requested offset.
         * @return The element in the specified offset.
         */
        __host__ __device__ inline uint8_t operator[](ptrdiff_t offset) const
        {
            return encoder::access(*this, offset);
        }

        /**
         * Retrieves an encoded character block from sequence.
         * @param offset The index of the requested block.
         * @return The requested block.
         */
        __host__ __device__ inline encoder::EncodedBlock getBlock(ptrdiff_t offset) const
        {
            return Buffer<encoder::EncodedBlock>::operator[](offset);
        }

        /**
         * Informs the length of the sequence.
         * @return The sequence's length.
         */
        __host__ __device__ inline size_t getLength() const
        {
            return this->getSize() * encoder::batchSize;
        }

        /**
         * Transforms the sequence into a string.
         * @return The sequence representation as a string.
         */
        inline std::string toString() const
        {
            return encoder::decode(*this);
        }
};

#endif