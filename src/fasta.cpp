/**
 * Multiple Sequence Alignment fasta file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <fstream>
#include <cstring>
#include <string>

#include "msa.hpp"
#include "fasta.hpp"
#include "cluster.hpp"

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
 * @param fname The name of the file to be loaded.
 * @return The number of sequences loaded from file.
 */
uint16_t Fasta::load(const std::string& fname)
{
    onlymaster {
        std::fstream fastafile(fname, std::fstream::in);

        if(fastafile.fail()) {
            finalize(ErrorCode::InvalidFile);
        }

        __debugh("loading from %s", fname.c_str());

        while(!fastafile.eof() && !fastafile.fail())
            this->extract(fastafile);

        fastafile.close();        
    }

    if(nodeinfo.size > 1) {
        this->broadcast();
    }

    onlymaster {
        __debugh("loaded %u sequences", this->getCount());
    }

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

    while(fastafile.peek() != 0x3E && std::getline(fastafile, buffer) && buffer.size() > 0)
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

/**
 * Sends the sequences loaded by the master node to all other nodes.
 * This method will send all sequences to all nodes.
 */
void Fasta::broadcast()
{
    uint16_t count = this->getCount();

    cluster::broadcast(&count);
    cluster::synchronize();

    uint32_t *sizes = new uint32_t [count];
    uint32_t szsum = 0;

    onlymaster {
        for(int i = 0; i < count; ++i)
            szsum += sizes[i] = this->list[i]->getLength();
    }

    cluster::broadcast(sizes, count);
    cluster::broadcast(&szsum);
    cluster::synchronize();

    char *data = new char [szsum];

    onlymaster {
        for(uint32_t i = 0, offset = 0; i < count; ++i) {
            memcpy(data + offset, this->list[i]->getBuffer(), sizeof(char) * sizes[i]);
            offset += sizes[i];
        }
    }

    cluster::broadcast(data, szsum);
    cluster::synchronize();

    onlyslaves {
        for(uint32_t i = 0, offset = 0; i < count; ++i) {
            this->push("__slave", data + offset, sizes[i]);
            offset += sizes[i];
        }
    }

    delete[] sizes;
    delete[] data;
}
