/** 
 * Multiple Sequence Alignment sequence file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <cstdint>
#include <cstring>
#include <ostream>
#include <string>
#include <vector>

#include "sequence.hpp"

/**
 * Instantiates a new immutable sequence.
 * @param buffer The buffer of which data will be copied from.
 */
Sequence::Sequence(const Buffer& buffer)
{
    this->length = buffer.getLength();
    this->buffer = new char [this->length];
    memcpy(this->buffer, &buffer, sizeof(char) * this->length);
}

/**
 * Instantiates a new immutable sequence.
 * @param string The string containing this sequence's data.
 */
Sequence::Sequence(const std::string& string)
{
    this->length = string.size();
    this->buffer = new char [this->length];
    memcpy(this->buffer, string.c_str(), sizeof(char) * this->length);
}

/**
 * Instantiates a new immutable sequence.
 * @param buffer The buffer of which data will be copied from.
 * @param size The size of the buffer.
 */
Sequence::Sequence(const char *buffer, uint32_t size)
{
    this->length = size;
    this->buffer = new char [this->length];
    memcpy(this->buffer, buffer, sizeof(char) * this->length);
}

/**
 * Destroys a sequence instance.
 */
Sequence::~Sequence() noexcept
{
    delete[] this->buffer;
}

/**
 * Copy assignment operator.
 * @param buffer The buffer of which data will be copied from.
 */
Sequence& Sequence::operator= (const Buffer& buffer)
{
    delete[] this->buffer;

    this->length = buffer.length;
    this->buffer = new char [buffer.length];
    memcpy(this->buffer, buffer.buffer, sizeof(char) * this->length);

    return *this;
}

/**
 * Instantiates a new sequence list.
 * @param slist An array of sequences of which data will be copied from.
 * @param count The number of sequences in given array.
 */
SequenceList::SequenceList(const Buffer *slist, uint16_t count)
{
    for(uint16_t i = 0; i < count; ++i)
        this->list.push_back(new Sequence(slist[i]));
}

/**
 * Instantiates a new sequence list.
 * @param slist A vector of sequences of which data will be copied from.
 */
SequenceList::SequenceList(const std::vector<Buffer>& slist)
{
    for(const Buffer& target : slist)
        this->list.push_back(new Sequence(target));
}

/**
 * Instantiates a new sequence list based on a subset of a list.
 * @param slist An array of sequences of which data will be copied from.
 * @param selected The selected sequences from the list.
 */
SequenceList::SequenceList(const SequenceList& slist, const std::vector<uint16_t>& selected)
{
    for(uint16_t index : selected)
        this->list.push_back(new Sequence(slist[index]));
}

/**
 * Instantiates a new sequence list based on a subset of a list.
 * @param slist An array of sequences of which data will be copied from.
 * @param selected The selected sequences from the list.
 * @param count The number of sequences in the given list.
 */
SequenceList::SequenceList(const SequenceList& slist, const uint16_t *selected, uint16_t count)
{
    for(uint16_t i = 0; i < count; ++i)
        this->list.push_back(new Sequence(slist[selected[i]]));
}

/**
 * Destroys a sequence list instance.
 */
SequenceList::~SequenceList() noexcept
{
    for(Sequence *sequence : this->list)
        delete sequence;
}

/**
 * Pushes a new sequence into the list.
 * @param buffer The buffer from which a sequence will be copied from.
 */
void SequenceList::push(const Buffer& buffer)
{
    this->list.push_back(new Sequence(buffer));
}

/**
 * Pushes a new sequence into the list.
 * @param string A string that will originate a new sequence into the list.
 */
void SequenceList::push(const std::string& string)
{
    this->list.push_back(new Sequence(string));
}

/**
 * Pushes a new sequence into the list.
 * @param buffer The buffer from which a sequence will be copied from.
 * @param size The size of the given buffer.
 */
void SequenceList::push(const char *buffer, uint16_t size)
{
    this->list.push_back(new Sequence(buffer, size));
}

/**
 * Creates a new sequence list based on a selection of sequences.
 * @param selected The sequences' indices to be sent a new list.
 * @return The new list of selected sequences.
 */
SequenceList SequenceList::select(const std::vector<uint16_t>& selected) const
{
    return SequenceList(*this, selected);
}

/**
 * Creates a new sequence list based on a selection of sequences.
 * @param selected The sequences' indices to be sent a new list.
 * @param count The number of selected sequences.
 * @return The new list of selected sequences.
 */
SequenceList SequenceList::select(const uint16_t *selected, uint16_t count) const
{
    return SequenceList(*this, selected, count);
}

/**
 * Consolidates and compactates the sequence list.
 * @return The new consolidated sequence list.
 */
CompactSequenceList SequenceList::compact() const
{
    return CompactSequenceList(*this);
}

/**
 * Instantiates a new consolidated sequence list.
 * @param slist A list of sequences of which data will be copied from.
 */
CompactSequenceList::CompactSequenceList(const SequenceList& slist)
:   Sequence(CompactSequenceList::merge(slist))
,   ref(new Buffer [slist.getCount()])
,   count(slist.getCount())
{
    for(uint32_t i = 0, off = 0; i < this->count; ++i) {
        this->ref[i].buffer = this->buffer + off;
        off += this->ref[i].length = slist[i].getLength();
    }
}

/**
 * Instantiates a new consolidated sequence list.
 * @param slist An array of sequences of which data will be copied from.
 * @param count The number of sequences in array.
 */
CompactSequenceList::CompactSequenceList(const Buffer *slist, uint16_t count)
:   Sequence(CompactSequenceList::merge(slist, count))
,   ref(new Buffer [count])
,   count(count)
{
    this->init(slist);
}

/**
 * Instantiates a new consolidated sequence list.
 * @param slist A vector of sequences of which data will be copied from.
 */
CompactSequenceList::CompactSequenceList(const std::vector<Buffer>& slist)
:   Sequence(CompactSequenceList::merge(slist.data(), slist.size()))
,   ref(new Buffer [slist.size()])
,   count(slist.size())
{
    this->init(slist.data());
}

/**
 * Destroys a consolidated sequence list.
 */
CompactSequenceList::~CompactSequenceList()
{
    delete[] this->ref;
}

/**
 * Sets up the buffers responsible for keeping track of internal sequences.
 * @param slist The list of original sequences being consolidated.
 */
void CompactSequenceList::init(const Buffer *slist)
{
    for(uint32_t i = 0, off = 0; i < this->count; ++i) {
        this->ref[i].buffer = this->buffer + off;
        off += this->ref[i].length = slist[i].getLength();
    }
}

/**
 * Merges all sequences from the list into a single sequnces.
 * @param slist The list of original sequences to be merged.
 */
std::string CompactSequenceList::merge(const SequenceList& slist)
{
    std::string merged;
    uint16_t count = slist.getCount();

    for(uint16_t i = 0; i < count; ++i)
        merged.append(slist[i].getBuffer(), slist[i].getLength());

    return merged;
}

/**
 * Merges all sequences from the list into a single sequnces.
 * @param slist The list of original sequences to be merged.
 */
std::string CompactSequenceList::merge(const Buffer *slist, uint16_t count)
{
    std::string merged;

    for(uint16_t i = 0; i < count; ++i)
        merged.append(slist[i].getBuffer(), slist[i].getLength());

    return merged;
}

/**
 * This function allows buffers to be directly printed into a ostream instance.
 * @param os The output stream object.
 * @param buffer The buffer to print.
 */
std::ostream& operator<< (std::ostream& os, const Buffer& buffer)
{
    for(uint32_t i = 0; i < buffer.length; ++i)
        os << buffer.buffer[i];

    return os;
}