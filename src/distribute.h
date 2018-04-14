/** @file distribute.h
 * @brief Parallel Multiple Sequence Alignment distribute header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _DISTRIBUTE_H
#define _DISTRIBUTE_H

#ifdef __cplusplus
extern "C" {
#endif

/** @enum mpiflag_t
 * @brief Enumerates tags for messages between nodes.
 */
typedef enum {
    M_SIZE = 0xa0
,   M_DATA = 0xa1
} mpiflag_t;

extern void collect(int);
extern void distribute();

#ifdef __cplusplus
}
#endif

#endif