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

Sequence::Sequence(const Buffer& buffer)
{
    this->length = buffer.getLength();
    this->buffer = new char [this->length];
    memcpy(this->buffer, &buffer, sizeof(char) * this->length);
}

Sequence::Sequence(const std::string& string)
{
    this->length = string.size();
    this->buffer = new char [this->length];
    memcpy(this->buffer, string.c_str(), sizeof(char) * this->length);
}

Sequence::Sequence(const char *buffer, uint32_t size)
{
    this->length = size;
    this->buffer = new char [this->length];
    memcpy(this->buffer, buffer, sizeof(char) * this->length);
}

Sequence::~Sequence() noexcept
{
    delete[] this->buffer;
}

Sequence& Sequence::operator= (const Buffer& buffer)
{
    delete[] this->buffer;

    this->length = buffer.length;
    this->buffer = new char [buffer.length];
    memcpy(this->buffer, buffer.buffer, sizeof(char) * this->length);

    return *this;
}

std::ostream& operator<< (std::ostream& os, const Buffer& buffer)
{
    for(uint32_t i = 0; i < buffer.length; ++i)
        os << buffer.buffer[i];

    return os;
}

SequenceList::SequenceList(const Buffer *slist, uint16_t count)
{
    for(uint16_t i = 0; i < count; ++i)
        this->list.push_back(new Sequence(slist[i]));
}

SequenceList::SequenceList(const std::vector<Buffer>& slist)
{
    for(const Buffer& target : slist)
        this->list.push_back(new Sequence(target));
}

SequenceList::SequenceList(const SequenceList& slist, const std::vector<uint16_t>& selected)
{
    for(uint16_t index : selected)
        this->list.push_back(new Sequence(slist[index]));
}

SequenceList::SequenceList(const SequenceList& slist, const uint16_t *selected, uint16_t count)
{
    for(uint16_t i = 0; i < count; ++i)
        this->list.push_back(new Sequence(slist[selected[i]]));
}

SequenceList::~SequenceList() noexcept
{
    for(Sequence *sequence : this->list)
        delete sequence;
}

void SequenceList::push(const Buffer& buffer)
{
    this->list.push_back(new Sequence(buffer));
}

void SequenceList::push(const std::string& string)
{
    this->list.push_back(new Sequence(string));
}

void SequenceList::push(const char *buffer, uint16_t size)
{
    this->list.push_back(new Sequence(buffer, size));
}

SequenceList SequenceList::select(const std::vector<uint16_t>& selected) const
{
    return SequenceList(*this, selected);
}

SequenceList SequenceList::select(const uint16_t *selected, uint16_t count) const
{
    return SequenceList(*this, selected, count);
}

CompactSequenceList SequenceList::compact() const
{
    return CompactSequenceList(*this);
}

CompactSequenceList::CompactSequenceList(const SequenceList& slist)
    : Sequence(CompactSequenceList::merge(slist))
    , ref(new Buffer [slist.getCount()])
    , count(slist.getCount())
{
    for(uint32_t i = 0, off = 0; i < this->count; ++i) {
        this->ref[i].buffer = this->buffer + off;
        off += this->ref[i].length = slist[i].getLength();
    }
}

CompactSequenceList::CompactSequenceList(const Buffer *slist, uint16_t count)
    : Sequence(CompactSequenceList::merge(slist, count))
    , ref(new Buffer [count])
    , count(count)
{
    this->init(slist);
}

CompactSequenceList::CompactSequenceList(const std::vector<Buffer>& slist)
    : Sequence(CompactSequenceList::merge(slist.data(), slist.size()))
    , ref(new Buffer [slist.size()])
    , count(slist.size())
{
    this->init(slist.data());
}

CompactSequenceList::~CompactSequenceList()
{
    delete[] this->ref;
}

void CompactSequenceList::init(const Buffer *slist)
{
    for(uint32_t i = 0, off = 0; i < this->count; ++i) {
        this->ref[i].buffer = this->buffer + off;
        off += this->ref[i].length = slist[i].getLength();
    }
}

std::string CompactSequenceList::merge(const SequenceList& slist)
{
    std::string merged;
    uint16_t count = slist.getCount();

    for(uint16_t i = 0; i < count; ++i)
        merged.append(slist[i].getBuffer(), slist[i].getLength());

    return merged;
}

std::string CompactSequenceList::merge(const Buffer *slist, uint16_t count)
{
    std::string merged;

    for(uint16_t i = 0; i < count; ++i)
        merged.append(slist[i].getBuffer(), slist[i].getLength());

    return merged;
}
