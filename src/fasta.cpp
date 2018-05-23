/**
 * Multiple Sequence Alignment fasta file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <fstream>
#include <string>

#include "msa.hpp"
#include "fasta.hpp"

/**
 * Instantiates a new fasta sequence.
 * @param description The sequence description.
 * @param string The string containing this sequence's data.
 */
FastaSequence::FastaSequence(const std::string& description, const std::string& string)
:   Sequence(string)
,   description(description)
{}

/**
 * Instantiates a new fasta sequence.
 * @param description The sequence description.
 * @param buffer The buffer containing this sequence's data.
 */
FastaSequence::FastaSequence(const std::string& description, const Buffer<char>& buffer)
:   Sequence(buffer)
,   description(description)
{}

/**
 * Instantiates a new fasta sequence.
 * @param description The sequence description.
 * @param buffer The buffer containing this sequence's data.
 * @param size The buffer's size.
 */
FastaSequence::FastaSequence(const std::string& description, const char *buffer, uint32_t size)
:   Sequence(buffer, size)
,   description(description)
{}

/**
 * Destroys all sequences read from fasta file.
 */
Fasta::~Fasta()
{
    for(FastaSequence *sequence : this->list)
        delete sequence;
}

/**
 * Reads a file and allocates memory to all sequences contained in it.
 * @param fname The name of the file to be read.
 * @return The number of sequences read from file.
 */
uint16_t Fasta::read(const std::string& fname)
{
    std::fstream fastafile(fname, std::fstream::in);

    if(fastafile.fail()) {
        finalize(ErrorCode::InvalidFile);
    }

    __debugh("loading from %s", fname.c_str());

    while(!fastafile.eof() && !fastafile.fail())
        this->extract(fastafile);

    fastafile.close();

    __debugh("loaded %u sequences", this->getCount());
    return this->getCount();
}

/**
 * Extracts a sequence out of the file and puts it into a buffer.
 * @param ffile The file to read sequence from.
 * @param dest The destination address for the sequence.
 * @return Could a sequence be extracted?
 */
bool Fasta::extract(std::fstream& fastafile)
{
    std::string buffer, description, sequence;

    while(!fastafile.eof() && fastafile.peek() != 0x3E)
        // Ignore all characters until a '>' is reached.
        // Our sequence will always have a description.
        fastafile.get();

    if(fastafile.eof())
        return false;

    std::getline(fastafile, description);
    description.erase(0, 1);

    while(std::getline(fastafile, buffer) && buffer.size() > 0)
        sequence.append(buffer);

    this->push(description, sequence);

    return true;
}

/**
 * Pushes a new sequence into the list.
 * @param description The new sequence description.
 * @param string A string that will originate a new sequence into the list.
 */
void Fasta::push(const std::string& description, const std::string& sequence)
{
    this->list.push_back(new FastaSequence(description, sequence));
}

/**
 * Pushes a new sequence into the list.
 * @param description The new sequence description.
 * @param buffer The buffer that will originate a new sequence into the list.
 */
void Fasta::push(const std::string& description, const Buffer<char>& buffer)
{
    this->list.push_back(new FastaSequence(description, buffer));
}

/**
 * Pushes a new sequence into the list.
 * @param description The new sequence description.
 * @param buffer The buffer that will originate a new sequence into the list.
 * @param size The buffer's size.
 */
void Fasta::push(const std::string& description, const char *buffer, uint32_t size)
{
    this->list.push_back(new FastaSequence(description, buffer, size));
}


//static const unsigned char map[26] = {
  /* A     B     C     D     E     F     G     H     I     J     K     L     M */
  //  0x00, 0x14, 0x01, 0x06, 0x08, 0x0E, 0x03, 0x09, 0x0A, 0x15, 0x0C, 0x0B, 0x0D,
  /* N     O     P     Q     R     S     T     U     V     W     X     Y     Z */
  //  0x05, 0x17, 0x0F, 0x07, 0x04, 0x10, 0x02, 0x17, 0x13, 0x11, 0x17, 0x12, 0x16,
//};

/**
 * Instantiates a new sequence list.
 * @param slist An array of sequences of which data will be copied from.
 * @param count The number of sequences in given array.
 */
//SequenceList::SequenceList(const Buffer *slist, uint16_t count)
//{
//    for(uint16_t i = 0; i < count; ++i)
        //this->list.push_back(new Sequence(slist[i]));
//}

/**
 * Instantiates a new sequence list.
 * @param slist A vector of sequences of which data will be copied from.
 */
//SequenceList::SequenceList(const std::vector<Buffer>& slist)
//{
//    for(const Buffer& target : slist)
        //this->list.push_back(new Sequence(target));
//}

/**
 * Instantiates a new sequence list based on a subset of a list.
 * @param slist An array of sequences of which data will be copied from.
 * @param selected The selected sequences from the list.
 */
//SequenceList::SequenceList(const SequenceList& slist, const std::vector<uint16_t>& selected)
//{
//    for(uint16_t index : selected)
        //this->list.push_back(new Sequence(slist[index]));
//}

/**
 * Instantiates a new sequence list based on a subset of a list.
 * @param slist An array of sequences of which data will be copied from.
 * @param selected The selected sequences from the list.
 * @param count The number of sequences in the given list.
 */
//SequenceList::SequenceList(const SequenceList& slist, const uint16_t *selected, uint16_t count)
//{
//    for(uint16_t i = 0; i < count; ++i)
        //this->list.push_back(new Sequence(slist[selected[i]]));
//}

/**
 * Destroys a sequence list instance.
 */
//SequenceList::~SequenceList() noexcept
//{
//    for(Sequence *sequence : this->list)
        //delete sequence;
//}

/**
 * Pushes a new sequence into the list.
 * @param buffer The buffer from which a sequence will be copied from.
 */
//void SequenceList::push(const Buffer& buffer)
//{
//    this->list.push_back(new Sequence(buffer));
//}
//
/**
 * Pushes a new sequence into the list.
 * @param string A string that will originate a new sequence into the list.
 */
//void SequenceList::push(const std::string& string)
//{
//    this->list.push_back(new Sequence(string));
//}
//
/**
 * Pushes a new sequence into the list.
 * @param buffer The buffer from which a sequence will be copied from.
 * @param size The size of the given buffer.
 */
//void SequenceList::push(const char *buffer, uint16_t size)
//{
//    this->list.push_back(new Sequence(buffer, size));
//}
//
/**
 * Creates a new sequence list based on a selection of sequences.
 * @param selected The sequences' indices to be sent a new list.
 * @return The new list of selected sequences.
 */
//SequenceList SequenceList::select(const std::vector<uint16_t>& selected) const
//{
//    //return SequenceList(*this, selected);
////}
////
//
/**
 * Creates a new sequence list based on a selection of sequences.
 * @param selected The sequences' indices to be sent a new list.
 * @param count The number of selected sequences.
 * @return The new list of selected sequences.
 */
//SequenceList SequenceList::select(const uint16_t *selected, uint16_t count) const
//{
//    //return SequenceList(*this, selected, count);
////}
////
//
/**
 * Consolidates and compactates the sequence list.
 * @return The new consolidated sequence list.
 */
//CompactSequenceList SequenceList::compact() const
//{
//    //return CompactSequenceList(*this);
////}
////
//
/**
 * Instantiates a new consolidated sequence list.
 * @param slist A list of sequences of which data will be copied from.
 */
//CompactSequenceList::CompactSequenceList(const SequenceList& slist)
////:   Sequence(CompactSequenceList::merge(slist))
////,   ref(new Buffer [slist.getCount()])
////,   count(slist.getCount())
////{
    //for(uint32_t i = 0, off = 0; i < this->count; ++i) {
        //this->ref[i].buffer = this->buffer + off;
        //off += this->ref[i].length = slist[i].getLength();
    //}
//}

/**
 * Instantiates a new consolidated sequence list.
 * @param slist An array of sequences of which data will be copied from.
 * @param count The number of sequences in array.
 */
//CompactSequenceList::CompactSequenceList(const Buffer *slist, uint16_t count)
////:   Sequence(CompactSequenceList::merge(slist, count))
////,   ref(new Buffer [count])
////,   count(count)
////{
    //this->init(slist);
//}

/**
 * Instantiates a new consolidated sequence list.
 * @param slist A vector of sequences of which data will be copied from.
 */
//CompactSequenceList::CompactSequenceList(const std::vector<Buffer>& slist)
////:   Sequence(CompactSequenceList::merge(slist.data(), slist.size()))
////,   ref(new Buffer [slist.size()])
////,   count(slist.size())
////{
    //this->init(slist.data());
//}

/**
 * Destroys a consolidated sequence list.
 */
//CompactSequenceList::~CompactSequenceList()
//{
//    delete[] this->ref;
//}
//
/**
 * Sets up the buffers responsible for keeping track of internal sequences.
 * @param slist The list of original sequences being consolidated.
 */
//void CompactSequenceList::init(const Buffer *slist)
//{
//    for(uint32_t i = 0, off = 0; i < this->count; ++i) {
        //this->ref[i].buffer = this->buffer + off;
        //off += this->ref[i].length = slist[i].getLength();
   // }
//}

/**
 * Merges all sequences from the list into a single sequnces.
 * @param slist The list of original sequences to be merged.
 */
//std::string CompactSequenceList::merge(const SequenceList& slist)
//{
//    std::string merged;
    //uint16_t count = slist.getCount();
//
    //for(uint16_t i = 0; i < count; ++i)
    //    merged.append(slist[i].getBuffer(), slist[i].getLength());

  //  return merged;
//}

/**
 * Merges all sequences from the list into a single sequnces.
 * @param slist The list of original sequences to be merged.
 */
//std::string CompactSequenceList::merge(const Buffer *slist, uint16_t count)
//{
//    std::string merged;
//
    //for(uint16_t i = 0; i < count; ++i)
    //    merged.append(slist[i].getBuffer(), slist[i].getLength());

  //  return merged;
//}
