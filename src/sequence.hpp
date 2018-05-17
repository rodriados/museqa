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

class Buffer
{
    protected:
        char *buffer = nullptr;
        uint32_t length = 0;

    public:
        inline char *getBuffer() const {
            return this->buffer;
        }

        inline uint32_t getLength() const {
            return this->length;
        }

        inline char *operator& () const {
            return this->buffer;
        }

        inline char& operator[] (uint32_t offset) const {
            return this->buffer[offset];
        }

    friend class Sequence;
    friend class CompactSequenceList;
    friend std::ostream& operator<< (std::ostream&, const Buffer&);
};

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


        inline uint16_t getCount() const {
            return this->list.size();
        }

        inline Sequence& operator[] (uint16_t offset) const {
            return *(this->list.at(offset));
        }

        void push(const Buffer&);
        void push(const std::string&);
        void push(const char *, uint16_t);

        SequenceList select(const std::vector<uint16_t>&) const;
        SequenceList select(const uint16_t *, uint16_t) const;
        class CompactSequenceList compact() const;
};

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

        inline uint16_t getCount() const {
            return this->count;
        }

        inline Buffer& operator[] (uint16_t offset) const {
            return this->ref[offset];
        }

    protected:
        void init(const Buffer *);

        static std::string merge(const SequenceList&);
        static std::string merge(const Buffer *, uint16_t);
};

#endif