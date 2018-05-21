/** 
 * Multiple Sequence Alignment sequence header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _SEQUENCE_HPP_
#define _SEQUENCE_HPP_

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

/**
 * Creates a general-purpose buffer. This buffer should only be written via
 * inherited classes' instances.
 * @since 0.1.alpha
 */
class Buffer
{
    protected:
        char *buffer = nullptr;
        uint32_t length = 0;

    public:
        /**
         * Gives access to buffer's data.
         * @return Buffer's pointer.
         */
        inline char *getBuffer() const
        {
            return this->buffer;
        }

        /**
         * Informs the length of data stored in buffer.
         * @return Buffer's length.
         */
        inline uint32_t getLength() const
        {
            return this->length;
        }

        /**
         * Gives access to buffer's data.
         * @return Buffer's pointer.
         */
        inline char *operator& () const
        {
            return this->buffer;
        }

        /**
         * Gives access to a specific location in buffer's data.
         * @return Buffer's position pointer.
         */
        inline char& operator[] (uint32_t offset) const
        {
            return this->buffer[offset];
        }

    friend class Sequence;
    friend class CompactSequenceList;
    friend std::ostream& operator<< (std::ostream&, const Buffer&);
};

/**
 * Creates an immutable sequence. This sequence is a buffer that should not be
 * changed after its instantiation.
 * @since 0.1.alpha
 */
class Sequence : public Buffer
{
    protected:
        Sequence() = default;

    public:
        Sequence(const Buffer&);
        Sequence(const std::string&);
        Sequence(const char *, uint32_t);

        ~Sequence() noexcept;

        Sequence& operator= (const Buffer&);
};

/**
 * Creates a sequence list. This sequence list is responsible for keeping
 * track of sequences within its scope. Once a sequence is put into the
 * list, it cannot leave.
 * @since 0.1.alpha
 */
class SequenceList
{
    protected:
        std::vector<Sequence *> list;

    public:
        SequenceList() = default;
        SequenceList(const Buffer *, uint16_t);
        SequenceList(const std::vector<Buffer>&);

        SequenceList(const SequenceList&, const std::vector<uint16_t>&);
        SequenceList(const SequenceList&, const uint16_t *, uint16_t);

        ~SequenceList() noexcept;

        /**
         * Informs the number of sequences in the list.
         * @return The list's number of sequences.
         */
        inline uint16_t getCount() const
        {
            return this->list.size();
        }

        /**
         * Gives access to a specific sequence of the list.
         * @return The requested sequence.
         */
        inline Sequence& operator[] (uint16_t offset) const
        {
            return *(this->list.at(offset));
        }

        void push(const Buffer&);
        void push(const std::string&);
        void push(const char *, uint16_t);

        SequenceList select(const std::vector<uint16_t>&) const;
        SequenceList select(const uint16_t *, uint16_t) const;
        class CompactSequenceList compact() const;
};

/**
 * Creates a compact sequence list. This immutable sequence list keeps all
 * of its sequences together in memory. This is useful not to worry about
 * moving these sequences separately.
 * @since 0.1.alpha
 */
class CompactSequenceList : public Sequence
{
    protected:
        Buffer *ref = nullptr;
        uint16_t count = 0;

    public:
        CompactSequenceList() = default;
        CompactSequenceList(const SequenceList&);
        CompactSequenceList(const Buffer *, uint16_t);
        CompactSequenceList(const std::vector<Buffer>&);

        ~CompactSequenceList() noexcept;

        /**
         * Informs the number of sequences in the list.
         * @return The list's number of sequences.
         */
        inline uint16_t getCount() const
        {
            return this->count;
        }

        /**
         * Gives access to a specific sequence of the list.
         * @return The requested sequence.
         */
        inline Buffer& operator[] (uint16_t offset) const
        {
            return this->ref[offset];
        }

    protected:
        void init(const Buffer *);

        static std::string merge(const SequenceList&);
        static std::string merge(const Buffer *, uint16_t);
};

#endif