/**
 * Multiple Sequence Alignment fasta header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _FASTA_HPP_
#define _FASTA_HPP_

#include <fstream>
#include <string>

#include "sequence.hpp"

/**
 * Represents a sequence read from a fasta file.
 * @since 0.1.alpha
 */
class FastaSequence : public Sequence
{
    private:
        const std::string description;      /// The sequence description.

    public:
        FastaSequence(const std::string&, const std::string&);
        FastaSequence(const std::string&, const Buffer<char>&);
        FastaSequence(const std::string&, const char *, uint32_t);
        virtual ~FastaSequence() noexcept = default;

        /**
         * Retrieves the description linked to the sequence.
         * @return The sequence's description.
         */
        inline const std::string& getDescription() const
        {
            return this->description;
        }
};

/**
 * Creates a list of sequences read from a fasta file. This sequence list
 * is responsible for keeping track of sequences within its scope. Once a
 * sequence is put into the list, it cannot leave.
 * @since 0.1.alpha
 */
class Fasta final
{
    protected:
        std::vector<FastaSequence *> list;  /// The list of sequences read from file.

    public:
        Fasta() = default;
        ~Fasta() noexcept;

        /**
         * Gives access to a specific sequence of the list.
         * @return The requested sequence.
         */
        inline const FastaSequence& operator[](uint16_t offset) const
        {
            return *(this->list.at(offset));
        }

        /**
         * Informs the number of sequences in the list.
         * @return The list's number of sequences.
         */
        inline uint16_t getCount() const
        {
            return this->list.size();
        }

        uint16_t read(const std::string&);

    private:
        bool extract(std::fstream&);
        void push(const std::string&, const std::string&);
        void push(const std::string&, const Buffer<char>&);
        void push(const std::string&, const char *, uint32_t);
};


/**
 * Creates a sequence list. This sequence list is responsible for keeping
 * track of sequences within its scope. Once a sequence is put into the
 * list, it cannot leave.
 * @since 0.1.alpha
 */
//class SequenceList
//{
  //  protected:
    //    std::vector<Sequence *> list;

    //public:
      //  SequenceList() = default;
        //SequenceList(const Buffer *, uint16_t);
        //SequenceList(const std::vector<Buffer>&);

//        SequenceList(const SequenceList&, const std::vector<uint16_t>&);
  //      SequenceList(const SequenceList&, const uint16_t *, uint16_t);

    //    ~SequenceList() noexcept;

        /**
         * Informs the number of sequences in the list.
         * @return The list's number of sequences.
         */
      //  inline uint16_t getCount() const
        //{
//            return this->list.size();
  //      }

        /**
         * Gives access to a specific sequence of the list.
         * @return The requested sequence.
         */
    //    inline const Sequence& operator[] (uint16_t offset) const
      //  {
        //    return *(this->list.at(offset));
//        }

  //      void push(const Buffer&);
    //    void push(const std::string&);
      //  void push(const char *, uint16_t);

//        SequenceList select(const std::vector<uint16_t>&) const;
  //      SequenceList select(const uint16_t *, uint16_t) const;
    //    class CompactSequenceList compact() const;
//};

/**
 * Creates a compact sequence list. This immutable sequence list keeps all
 * of its sequences together in memory. This is useful not to worry about
 * moving these sequences separately.
 * @since 0.1.alpha
 */
//class CompactSequenceList : public Sequence
//{
  //  protected:
    //    Buffer *ref = nullptr;
      //  uint16_t count = 0;

//    public:
  //      CompactSequenceList() = default;
    //    CompactSequenceList(const SequenceList&);
      //  CompactSequenceList(const Buffer *, uint16_t);
        //CompactSequenceList(const std::vector<Buffer>&);

//        ~CompactSequenceList() noexcept;

        /**
         * Informs the number of sequences in the list.
         * @return The list's number of sequences.
         */
  //      inline uint16_t getCount() const
    //    {
      //      return this->count;
        //}

        /**
         * Gives access to a specific sequence of the list.
         * @return The requested sequence.
         */
//        inline Buffer& operator[] (uint16_t offset) const
  //      {
    //        return this->ref[offset];
      //  }

//    protected:
  //      void init(const Buffer *);

    //    static std::string merge(const SequenceList&);
      //  static std::string merge(const Buffer *, uint16_t);
//};


#endif