/**
 * Multiple Sequence Alignment pairwise interface header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _PAIRWISE_CUH_
#define _PAIRWISE_CUH_

#include "pairwise/pairwise.cuh"

/*
 * Defining some configuration macros. These can be changed if needed.
 */
#define __pw_threads_per_block__ 32
#define __pw_prefer_shared_mem__ 0

/*
 * Exposes pairwise classes to the global namespace.
 * @since 0.1.alpha
 */
typedef pairwise::Score Score;
typedef pairwise::Pairwise Pairwise;

#endif








/** @class pairwise_t
 * @brief Stores data and structures needed for executing pairwise algorithm.
 * @var seqchar The pointer to character sequences.
 * @var nseq The number of sequences loaded from file.
 * @var npair The number of working pairs received to process.
 * @var seq The sequences' positions.
 * @var pair The working pairs to process.
 * @var score The score of each working pair processed.
 */
/*class pairwise_t
{
public:
    char *seqchar;
    uint16_t nseq;
    uint32_t npair;
    uint32_t clength;
    position_t *seq;
    workpair_t *pair;
    score_t *score;

public:
    pairwise_t();
    ~pairwise_t();

    void load(const fasta_t *);

    void daemon();
    void pairwise();

private:
    void generate();

    bool select(bool[], std::vector<uint32_t>&) const;
    void blosum(needleman_t&);
    void run(needleman_t&);

    void alloc(needleman_t&, std::vector<uint32_t>&);
    void allocseq(needleman_t&, std::vector<uint16_t>&);
    void request(std::vector<uint16_t>&);
    void destroy(needleman_t&) const;
};

namespace pairwise
{   
#ifdef __CUDACC__
    extern __global__ void needleman(needleman_t, score_t *);
#endif
}

namespace daemon
{
    enum tag_t {
        SYN = 0x11  // Initiates a new request
    ,   END         // Destroys a daemon thread
    ,   PLD         // The requested sequences payload
    ,   BSZ         // The buffer size response
    ,   CHR         // The characters response
    ,   POS         // The positions response
    };

    extern void run(const pairwise_t *, int);
    extern void response(const pairwise_t *, int, int, short[]);
    extern char *request(std::vector<uint16_t>&, position_t *, int&);
    extern void destroy();
}*/

/** @fn int divceil(int, int)
 * @brief Calculates the division between two numbers and rounds it up.
 * @param a The number to be divided.
 * @param b The number to divide by.
 * @return The resulting number.
 */
/*inline int divceil(int a, int b)
{
    return (a / b) + !!(a % b);
}*/

/** @fn int align(int, int)
 * @brief Calculates the alignment for a given size.
 * @param size The size to be aligned.
 * @param align The alignment to use for given size.
 * @return The new aligned size.
 */
/*inline int align(int size, int align = 4)
{
    return divceil(size, align) * align;
}

#endif*/