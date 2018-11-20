/**
 * Multiple Sequence Alignment blosum file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <cstdint>
#include <cstring>
#include <map>

#include "cli.hpp"
#include "device.cuh"
#include "pairwise/pairwise.hpp"

/*
 * The names of scoring tables. This will be used as available parameters available
 * when choosing a scoring table.
 */
static const std::vector<std::string> tablenames =
    {"blosum62", "blosum45", "blosum50", "blosum80", "blosum90", "pam250"};

/*
 * The scoring tables data. One of these tables will be transfered to device memory
 * so it can be used to score sequence alignments. The first table, index-zero, is
 * used as the default, when no valid parameter is found to indicate which table
 * should be used instead.
 */
static const int8_t tabledata[][25][25] = {
    {   /* blosum62 */
        /*A  C  T  G  R  N  D  Q  E  H  I  L  K  M  F  P  S  W  Y  V  B  J  Z  X  **/
        { 4, 0, 0, 0,-1,-2,-2,-1,-1,-2,-1,-1,-1,-1,-2,-1, 1,-3,-2, 0,-2,-1,-1,-1,-4}
    ,   { 0, 9,-1,-3,-3,-3,-3,-3,-4,-3,-1,-1,-3,-1,-2,-3,-1,-2,-2,-1,-3,-1,-3,-1,-4}
    ,   { 0,-1, 5,-2,-1, 0,-1,-1,-1,-2,-1,-1,-1,-1,-2,-1, 1,-2,-2, 0,-1,-1,-1,-1,-4}
    ,   { 0,-3,-2, 6,-2, 0,-1,-2,-2,-2,-4,-4,-2,-3,-3,-2, 0,-2,-3,-3,-1,-4,-2,-1,-4}
    ,   {-1,-3,-1,-2, 5, 0,-2, 1, 0, 0,-3,-2, 2,-1,-3,-2,-1,-3,-2,-3,-1,-2, 0,-1,-4}
    ,   {-2,-3, 0, 0, 0, 6, 1, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1,-4,-2,-3, 4,-3, 0,-1,-4}
    ,   {-2,-3,-1,-1,-2, 1, 6, 0, 2,-1,-3,-4,-1,-3,-3,-1, 0,-4,-3,-3, 4,-3, 1,-1,-4}
    ,   {-1,-3,-1,-2, 1, 0, 0, 5, 2, 0,-3,-2, 1, 0,-3,-1, 0,-2,-1,-2, 0,-2, 4,-1,-4}
    ,   {-1,-4,-1,-2, 0, 0, 2, 2, 5, 0,-3,-3, 1,-2,-3,-1, 0,-3,-2,-2, 1,-3, 4,-1,-4}
    ,   {-2,-3,-2,-2, 0, 1,-1, 0, 0, 8,-3,-3,-1,-2,-1,-2,-1,-2, 2,-3, 0,-3, 0,-1,-4}
    ,   {-1,-1,-1,-4,-3,-3,-3,-3,-3,-3, 4, 2,-3, 1, 0,-3,-2,-3,-1, 3,-3, 3,-3,-1,-4}
    ,   {-1,-1,-1,-4,-2,-3,-4,-2,-3,-3, 2, 4,-2, 2, 0,-3,-2,-2,-1, 1,-4, 3,-3,-1,-4}
    ,   {-1,-3,-1,-2, 2, 0,-1, 1, 1,-1,-3,-2, 5,-1,-3,-1, 0,-3,-2,-2, 0,-3, 1,-1,-4}
    ,   {-1,-1,-1,-3,-1,-2,-3, 0,-2,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1, 1,-3, 2,-1,-1,-4}
    ,   {-2,-2,-2,-3,-3,-3,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2, 1, 3,-1,-3, 0,-3,-1,-4}
    ,   {-1,-3,-1,-2,-2,-2,-1,-1,-1,-2,-3,-3,-1,-2,-4, 7,-1,-4,-3,-2,-2,-3,-1,-1,-4}
    ,   { 1,-1, 1, 0,-1, 1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4,-3,-2,-2, 0,-2, 0,-1,-4}
    ,   {-3,-2,-2,-2,-3,-4,-4,-2,-3,-2,-3,-2,-3,-1, 1,-4,-3,11, 2,-3,-4,-2,-2,-1,-4}
    ,   {-2,-2,-2,-3,-2,-2,-3,-1,-2, 2,-1,-1,-2,-1, 3,-3,-2, 2, 7,-1,-3,-1,-2,-1,-4}
    ,   { 0,-1, 0,-3,-3,-3,-3,-2,-2,-3, 3, 1,-2, 1,-1,-2,-2,-3,-1, 4,-3, 2,-2,-1,-4}
    ,   {-2,-3,-1,-1,-1, 4, 4, 0, 1, 0,-3,-4, 0,-3,-3,-2, 0,-4,-3,-3, 4,-3, 0,-1,-4}
    ,   {-1,-1,-1,-4,-2,-3,-3,-2,-3,-3, 3, 3,-3, 2, 0,-3,-2,-2,-1, 2,-3, 3,-3,-1,-4}
    ,   {-1,-3,-1,-2, 0, 0, 1, 4, 4, 0,-3,-3, 1,-1,-3,-1, 0,-2,-2,-2, 0,-3, 4,-1,-4}
    ,   {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-4}
    ,   {-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4, 0}
    }
,   {   /* blosum45 */
        /*A  C  T  G  R  N  D  Q  E  H  I  L  K  M  F  P  S  W  Y  V  B  J  Z  X  **/
        { 5,-1, 0,-1,-2,-1,-2,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1,-2,-2, 0,-1,-1,-1,-1,-5}
    ,   {-1,12,-1,-3,-3,-2,-3,-3,-3,-3,-3,-2,-3,-2,-2,-4,-1,-5,-3,-1,-2,-2,-3,-1,-5}
    ,   { 0,-1, 5,-1,-1, 0,-1,-1,-2,-2,-1,-1,-1,-1,-1,-1, 2,-3,-1, 0, 0,-1,-1,-1,-5}
    ,   {-1,-3,-1, 6, 0, 0, 2, 2,-2, 0,-3,-2, 1,-2,-3, 0, 0,-3,-2,-3, 1,-3, 5,-1,-5}
    ,   {-2,-3,-1, 0, 7, 0,-1, 1,-2, 0,-3,-2, 3,-1,-2,-2,-1,-2,-1,-2,-1,-3, 1,-1,-5}
    ,   {-1,-2, 0, 0, 0, 6, 2, 0, 0, 1,-2,-3, 0,-2,-2,-2, 1,-4,-2,-3, 5,-3, 0,-1,-5}
    ,   {-2,-3,-1, 2,-1, 2, 7, 0,-1, 0,-4,-3, 0,-3,-4,-1, 0,-4,-2,-3, 6,-3, 1,-1,-5}
    ,   {-1,-3,-1, 2, 1, 0, 0, 6,-2, 1,-2,-2, 1, 0,-4,-1, 0,-2,-1,-3, 0,-2, 4,-1,-5}
    ,   { 0,-3,-2,-2,-2, 0,-1,-2, 7,-2,-4,-3,-2,-2,-3,-2, 0,-2,-3,-3,-1,-4,-2,-1,-5}
    ,   {-2,-3,-2, 0, 0, 1, 0, 1,-2,10,-3,-2,-1, 0,-2,-2,-1,-3, 2,-3, 0,-2, 0,-1,-5}
    ,   {-1,-3,-1,-3,-3,-2,-4,-2,-4,-3, 5, 2,-3, 2, 0,-2,-2,-2, 0, 3,-3, 4,-3,-1,-5}
    ,   {-1,-2,-1,-2,-2,-3,-3,-2,-3,-2, 2, 5,-3, 2, 1,-3,-3,-2, 0, 1,-3, 4,-2,-1,-5}
    ,   {-1,-3,-1, 1, 3, 0, 0, 1,-2,-1,-3,-3, 5,-1,-3,-1,-1,-2,-1,-2, 0,-3, 1,-1,-5}
    ,   {-1,-2,-1,-2,-1,-2,-3, 0,-2, 0, 2, 2,-1, 6, 0,-2,-2,-2, 0, 1,-2, 2,-1,-1,-5}
    ,   {-2,-2,-1,-3,-2,-2,-4,-4,-3,-2, 0, 1,-3, 0, 8,-3,-2, 1, 3, 0,-3, 1,-3,-1,-5}
    ,   {-1,-4,-1, 0,-2,-2,-1,-1,-2,-2,-2,-3,-1,-2,-3, 9,-1,-3,-3,-3,-2,-3,-1,-1,-5}
    ,   { 1,-1, 2, 0,-1, 1, 0, 0, 0,-1,-2,-3,-1,-2,-2,-1, 4,-4,-2,-1, 0,-2, 0,-1,-5}
    ,   {-2,-5,-3,-3,-2,-4,-4,-2,-2,-3,-2,-2,-2,-2, 1,-3,-4,15, 3,-3,-4,-2,-2,-1,-5}
    ,   {-2,-3,-1,-2,-1,-2,-2,-1,-3, 2, 0, 0,-1, 0, 3,-3,-2, 3, 8,-1,-2, 0,-2,-1,-5}
    ,   { 0,-1, 0,-3,-2,-3,-3,-3,-3,-3, 3, 1,-2, 1, 0,-3,-1,-3,-1, 5,-3, 2,-3,-1,-5}
    ,   {-1,-2, 0, 1,-1, 5, 6, 0,-1, 0,-3,-3, 0,-2,-3,-2, 0,-4,-2,-3, 5,-3, 1,-1,-5}
    ,   {-1,-2,-1,-3,-3,-3,-3,-2,-4,-2, 4, 4,-3, 2, 1,-3,-2,-2, 0, 2,-3, 4,-2,-1,-5}
    ,   {-1,-3,-1, 5, 1, 0, 1, 4,-2, 0,-3,-2, 1,-1,-3,-1, 0,-2,-2,-3, 1,-2, 5,-1,-5}
    ,   {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-5}
    ,   {-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5, 0}
    }
,   {   /* blosum50 */
        /*A  C  T  G  R  N  D  Q  E  H  I  L  K  M  F  P  S  W  Y  V  B  J  Z  X  **/
        { 5,-1,-3,-2,-2,-1,-2,-1,-1, 0,-1,-2,-1,-1,-3,-1, 1, 0,-2, 0,-2,-2,-1,-1,-5}
    ,   {-1,13,-5,-3,-4,-2,-4,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-3,-1,-3,-2,-3,-1,-5}
    ,   {-3,-5,15,-3,-3,-4,-5,-1,-3,-3,-3,-2,-3,-1, 1,-4,-4,-3, 2,-3,-5,-2,-2,-1,-5}
    ,   {-2,-3,-3,10, 0, 1,-1, 1, 0,-2,-4,-3, 0,-1,-1,-2,-1,-2, 2,-4, 0,-3, 0,-1,-5}
    ,   {-2,-4,-3, 0, 7,-1,-2, 1, 0,-3,-4,-3, 3,-2,-3,-3,-1,-1,-1,-3,-1,-3, 0,-1,-5}
    ,   {-1,-2,-4, 1,-1, 7, 2, 0, 0, 0,-3,-4, 0,-2,-4,-2, 1, 0,-2,-3, 5,-4, 0,-1,-5}
    ,   {-2,-4,-5,-1,-2, 2, 8, 0, 2,-1,-4,-4,-1,-4,-5,-1, 0,-1,-3,-4, 6,-4, 1,-1,-5}
    ,   {-1,-3,-1, 1, 1, 0, 0, 7, 2,-2,-3,-2, 2, 0,-4,-1, 0,-1,-1,-3, 0,-3, 4,-1,-5}
    ,   {-1,-3,-3, 0, 0, 0, 2, 2, 6,-3,-4,-3, 1,-2,-3,-1,-1,-1,-2,-3, 1,-3, 5,-1,-5}
    ,   { 0,-3,-3,-2,-3, 0,-1,-2,-3, 8,-4,-4,-2,-3,-4,-2, 0,-2,-3,-4,-1,-4,-2,-1,-5}
    ,   {-1,-2,-3,-4,-4,-3,-4,-3,-4,-4, 5, 2,-3, 2, 0,-3,-3,-1,-1, 4,-4, 4,-3,-1,-5}
    ,   {-2,-2,-2,-3,-3,-4,-4,-2,-3,-4, 2, 5,-3, 3, 1,-4,-3,-1,-1, 1,-4, 4,-3,-1,-5}
    ,   {-1,-3,-3, 0, 3, 0,-1, 2, 1,-2,-3,-3, 6,-2,-4,-1, 0,-1,-2,-3, 0,-3, 1,-1,-5}
    ,   {-1,-2,-1,-1,-2,-2,-4, 0,-2,-3, 2, 3,-2, 7, 0,-3,-2,-1, 0, 1,-3, 2,-1,-1,-5}
    ,   {-3,-2, 1,-1,-3,-4,-5,-4,-3,-4, 0, 1,-4, 0, 8,-4,-3,-2, 4,-1,-4, 1,-4,-1,-5}
    ,   {-1,-4,-4,-2,-3,-2,-1,-1,-1,-2,-3,-4,-1,-3,-4,10,-1,-1,-3,-3,-2,-3,-1,-1,-5}
    ,   { 1,-1,-4,-1,-1, 1, 0, 0,-1, 0,-3,-3, 0,-2,-3,-1, 5, 2,-2,-2, 0,-3, 0,-1,-5}
    ,   { 0,-1,-3,-2,-1, 0,-1,-1,-1,-2,-1,-1,-1,-1,-2,-1, 2, 5,-2, 0, 0,-1,-1,-1,-5}
    ,   {-2,-3, 2, 2,-1,-2,-3,-1,-2,-3,-1,-1,-2, 0, 4,-3,-2,-2, 8,-1,-3,-1,-2,-1,-5}
    ,   { 0,-1,-3,-4,-3,-3,-4,-3,-3,-4, 4, 1,-3, 1,-1,-3,-2, 0,-1, 5,-3, 2,-3,-1,-5}
    ,   {-2,-3,-5, 0,-1, 5, 6, 0, 1,-1,-4,-4, 0,-3,-4,-2, 0, 0,-3,-3, 6,-4, 1,-1,-5}
    ,   {-2,-2,-2,-3,-3,-4,-4,-3,-3,-4, 4, 4,-3, 2, 1,-3,-3,-1,-1, 2,-4, 4,-3,-1,-5}
    ,   {-1,-3,-2, 0, 0, 0, 1, 4, 5,-2,-3,-3, 1,-1,-4,-1, 0,-1,-2,-3, 1,-3, 5,-1,-5}
    ,   {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-5}
    ,   {-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5, 0}
    }
,   {   /* blosum80 */
        /*A  C  T  G  R  N  D  Q  E  H  I  L  K  M  F  P  S  W  Y  V  B  J  Z  X  **/
        { 5,-1, 0,-1,-2,-2,-2,-1, 0,-2,-2,-2,-1,-1,-3,-1, 1,-3,-2, 0,-2,-2,-1,-1,-6}
    ,   {-1, 9,-1,-5,-4,-3,-4,-4,-4,-4,-2,-2,-4,-2,-3,-4,-2,-3,-3,-1,-4,-2,-4,-1,-6}
    ,   { 0,-1, 5,-1,-1, 0,-1,-1,-2,-2,-1,-2,-1,-1,-2,-2, 1,-4,-2, 0,-1,-1,-1,-1,-6}
    ,   {-1,-5,-1, 6,-1,-1, 1, 2,-3, 0,-4,-4, 1,-2,-4,-2, 0,-4,-3,-3, 1,-4, 5,-1,-6}
    ,   {-2,-4,-1,-1, 6,-1,-2, 1,-3, 0,-3,-3, 2,-2,-4,-2,-1,-4,-3,-3,-1,-3, 0,-1,-6}
    ,   {-2,-3, 0,-1,-1, 6, 1, 0,-1, 0,-4,-4, 0,-3,-4,-3, 0,-4,-3,-4, 5,-4, 0,-1,-6}
    ,   {-2,-4,-1, 1,-2, 1, 6,-1,-2,-2,-4,-5,-1,-4,-4,-2,-1,-6,-4,-4, 5,-5, 1,-1,-6}
    ,   {-1,-4,-1, 2, 1, 0,-1, 6,-2, 1,-3,-3, 1, 0,-4,-2, 0,-3,-2,-3, 0,-3, 4,-1,-6}
    ,   { 0,-4,-2,-3,-3,-1,-2,-2, 6,-3,-5,-4,-2,-4,-4,-3,-1,-4,-4,-4,-1,-5,-3,-1,-6}
    ,   {-2,-4,-2, 0, 0, 0,-2, 1,-3, 8,-4,-3,-1,-2,-2,-3,-1,-3, 2,-4,-1,-4, 0,-1,-6}
    ,   {-2,-2,-1,-4,-3,-4,-4,-3,-5,-4, 5, 1,-3, 1,-1,-4,-3,-3,-2, 3,-4, 3,-4,-1,-6}
    ,   {-2,-2,-2,-4,-3,-4,-5,-3,-4,-3, 1, 4,-3, 2, 0,-3,-3,-2,-2, 1,-4, 3,-3,-1,-6}
    ,   {-1,-4,-1, 1, 2, 0,-1, 1,-2,-1,-3,-3, 5,-2,-4,-1,-1,-4,-3,-3,-1,-3, 1,-1,-6}
    ,   {-1,-2,-1,-2,-2,-3,-4, 0,-4,-2, 1, 2,-2, 6, 0,-3,-2,-2,-2, 1,-3, 2,-1,-1,-6}
    ,   {-3,-3,-2,-4,-4,-4,-4,-4,-4,-2,-1, 0,-4, 0, 6,-4,-3, 0, 3,-1,-4, 0,-4,-1,-6}
    ,   {-1,-4,-2,-2,-2,-3,-2,-2,-3,-3,-4,-3,-1,-3,-4, 8,-1,-5,-4,-3,-2,-4,-2,-1,-6}
    ,   { 1,-2, 1, 0,-1, 0,-1, 0,-1,-1,-3,-3,-1,-2,-3,-1, 5,-4,-2,-2, 0,-3, 0,-1,-6}
    ,   {-3,-3,-4,-4,-4,-4,-6,-3,-4,-3,-3,-2,-4,-2, 0,-5,-4,11, 2,-3,-5,-3,-3,-1,-6}
    ,   {-2,-3,-2,-3,-3,-3,-4,-2,-4, 2,-2,-2,-3,-2, 3,-4,-2, 2, 7,-2,-3,-2,-3,-1,-6}
    ,   { 0,-1, 0,-3,-3,-4,-4,-3,-4,-4, 3, 1,-3, 1,-1,-3,-2,-3,-2, 4,-4, 2,-3,-1,-6}
    ,   {-2,-4,-1, 1,-1, 5, 5, 0,-1,-1,-4,-4,-1,-3,-4,-2, 0,-5,-3,-4, 5,-4, 0,-1,-6}
    ,   {-2,-2,-1,-4,-3,-4,-5,-3,-5,-4, 3, 3,-3, 2, 0,-4,-3,-3,-2, 2,-4, 3,-3,-1,-6}
    ,   {-1,-4,-1, 5, 0, 0, 1, 4,-3, 0,-4,-3, 1,-1,-4,-2, 0,-3,-3,-3, 0,-3, 5,-1,-6}
    ,   {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-6}
    ,   {-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6, 0}
    }
,   {   /* blosum90 */
        /*A  C  T  G  R  N  D  Q  E  H  I  L  K  M  F  P  S  W  Y  V  B  J  Z  X  **/
        { 5,-1, 0,-1,-2,-2,-3,-1, 0,-2,-2,-2,-1,-2,-3,-1, 1,-4,-3,-1,-2,-2,-1,-1,-6}
    ,   {-1, 9,-2,-6,-5,-4,-5,-4,-4,-5,-2,-2,-4,-2,-3,-4,-2,-4,-4,-2,-4,-2,-5,-1,-6}
    ,   { 0,-2, 6,-1,-2, 0,-2,-1,-3,-2,-1,-2,-1,-1,-3,-2, 1,-4,-2,-1,-1,-2,-1,-1,-6}
    ,   {-1,-6,-1, 6,-1,-1, 1, 2,-3,-1,-4,-4, 0,-3,-5,-2,-1,-5,-4,-3, 1,-4, 5,-1,-6}
    ,   {-2,-5,-2,-1, 6,-1,-3, 1,-3, 0,-4,-3, 2,-2,-4,-3,-1,-4,-3,-3,-2,-3, 0,-1,-6}
    ,   {-2,-4, 0,-1,-1, 7, 1, 0,-1, 0,-4,-4, 0,-3,-4,-3, 0,-5,-3,-4, 5,-4,-1,-1,-6}
    ,   {-3,-5,-2, 1,-3, 1, 7,-1,-2,-2,-5,-5,-1,-4,-5,-3,-1,-6,-4,-5, 5,-5, 1,-1,-6}
    ,   {-1,-4,-1, 2, 1, 0,-1, 7,-3, 1,-4,-3, 1, 0,-4,-2,-1,-3,-3,-3,-1,-3, 5,-1,-6}
    ,   { 0,-4,-3,-3,-3,-1,-2,-3, 6,-3,-5,-5,-2,-4,-5,-3,-1,-4,-5,-5,-2,-5,-3,-1,-6}
    ,   {-2,-5,-2,-1, 0, 0,-2, 1,-3, 8,-4,-4,-1,-3,-2,-3,-2,-3, 1,-4,-1,-4, 0,-1,-6}
    ,   {-2,-2,-1,-4,-4,-4,-5,-4,-5,-4, 5, 1,-4, 1,-1,-4,-3,-4,-2, 3,-5, 3,-4,-1,-6}
    ,   {-2,-2,-2,-4,-3,-4,-5,-3,-5,-4, 1, 5,-3, 2, 0,-4,-3,-3,-2, 0,-5, 4,-4,-1,-6}
    ,   {-1,-4,-1, 0, 2, 0,-1, 1,-2,-1,-4,-3, 6,-2,-4,-2,-1,-5,-3,-3,-1,-3, 1,-1,-6}
    ,   {-2,-2,-1,-3,-2,-3,-4, 0,-4,-3, 1, 2,-2, 7,-1,-3,-2,-2,-2, 0,-4, 2,-2,-1,-6}
    ,   {-3,-3,-3,-5,-4,-4,-5,-4,-5,-2,-1, 0,-4,-1, 7,-4,-3, 0, 3,-2,-4, 0,-4,-1,-6}
    ,   {-1,-4,-2,-2,-3,-3,-3,-2,-3,-3,-4,-4,-2,-3,-4, 8,-2,-5,-4,-3,-3,-4,-2,-1,-6}
    ,   { 1,-2, 1,-1,-1, 0,-1,-1,-1,-2,-3,-3,-1,-2,-3,-2, 5,-4,-3,-2, 0,-3,-1,-1,-6}
    ,   {-4,-4,-4,-5,-4,-5,-6,-3,-4,-3,-4,-3,-5,-2, 0,-5,-4,11, 2,-3,-6,-3,-4,-1,-6}
    ,   {-3,-4,-2,-4,-3,-3,-4,-3,-5, 1,-2,-2,-3,-2, 3,-4,-3, 2, 8,-3,-4,-2,-3,-1,-6}
    ,   {-1,-2,-1,-3,-3,-4,-5,-3,-5,-4, 3, 0,-3, 0,-2,-3,-2,-3,-3, 5,-4, 1,-3,-1,-6}
    ,   {-2,-4,-1, 1,-2, 5, 5,-1,-2,-1,-5,-5,-1,-4,-4,-3, 0,-6,-4,-4, 5,-5, 0,-1,-6}
    ,   {-2,-2,-2,-4,-3,-4,-5,-3,-5,-4, 3, 4,-3, 2, 0,-4,-3,-3,-2, 1,-5, 4,-4,-1,-6}
    ,   {-1,-5,-1, 5, 0,-1, 1, 5,-3, 0,-4,-4, 1,-2,-4,-2,-1,-4,-3,-3, 0,-4, 5,-1,-6}
    ,   {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-6}
    ,   {-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6, 0}
    }
,   {   /* pam250 */
        /*A  C  T  G  R  N  D  Q  E  H  I  L  K  M  F  P  S  W  Y  V  B  J  Z  X  **/
        { 2,-2, 1, 0,-2, 0, 0, 0, 1,-1,-1,-2,-1,-1,-3, 1, 1,-6,-3, 0, 0,-1, 0,-1,-8}
    ,   {-2,12,-2,-5,-4,-4,-5,-5,-3,-3,-2,-6,-5,-5,-4,-3, 0,-8, 0,-2,-4,-5,-5,-1,-8}
    ,   { 1,-2, 3, 0,-1, 0, 0,-1, 0,-1, 0,-2, 0,-1,-3, 0, 1,-5,-3, 0, 0,-1,-1,-1,-8}
    ,   { 0,-5, 0, 4,-1, 1, 3, 2, 0, 1,-2,-3, 0,-2,-5,-1, 0,-7,-4,-2, 3,-3, 3,-1,-8}
    ,   {-2,-4,-1,-1, 6, 0,-1, 1,-3, 2,-2,-3, 3, 0,-4, 0, 0, 2,-4,-2,-1,-3, 0,-1,-8}
    ,   { 0,-4, 0, 1, 0, 2, 2, 1, 0, 2,-2,-3, 1,-2,-3, 0, 1,-4,-2,-2, 2,-3, 1,-1,-8}
    ,   { 0,-5, 0, 3,-1, 2, 4, 2, 1, 1,-2,-4, 0,-3,-6,-1, 0,-7,-4,-2, 3,-3, 3,-1,-8}
    ,   { 0,-5,-1, 2, 1, 1, 2, 4,-1, 3,-2,-2, 1,-1,-5, 0,-1,-5,-4,-2, 1,-2, 3,-1,-8}
    ,   { 1,-3, 0, 0,-3, 0, 1,-1, 5,-2,-3,-4,-2,-3,-5, 0, 1,-7,-5,-1, 0,-4, 0,-1,-8}
    ,   {-1,-3,-1, 1, 2, 2, 1, 3,-2, 6,-2,-2, 0,-2,-2, 0,-1,-3, 0,-2, 1,-2, 2,-1,-8}
    ,   {-1,-2, 0,-2,-2,-2,-2,-2,-3,-2, 5, 2,-2, 2, 1,-2,-1,-5,-1, 4,-2, 3,-2,-1,-8}
    ,   {-2,-6,-2,-3,-3,-3,-4,-2,-4,-2, 2, 6,-3, 4, 2,-3,-3,-2,-1, 2,-3, 5,-3,-1,-8}
    ,   {-1,-5, 0, 0, 3, 1, 0, 1,-2, 0,-2,-3, 5, 0,-5,-1, 0,-3,-4,-2, 1,-3, 0,-1,-8}
    ,   {-1,-5,-1,-2, 0,-2,-3,-1,-3,-2, 2, 4, 0, 6, 0,-2,-2,-4,-2, 2,-2, 3,-2,-1,-8}
    ,   {-3,-4,-3,-5,-4,-3,-6,-5,-5,-2, 1, 2,-5, 0, 9,-5,-3, 0, 7,-1,-4, 2,-5,-1,-8}
    ,   { 1,-3, 0,-1, 0, 0,-1, 0, 0, 0,-2,-3,-1,-2,-5, 6, 1,-6,-5,-1,-1,-2, 0,-1,-8}
    ,   { 1, 0, 1, 0, 0, 1, 0,-1, 1,-1,-1,-3, 0,-2,-3, 1, 2,-2,-3,-1, 0,-2, 0,-1,-8}
    ,   {-6,-8,-5,-7, 2,-4,-7,-5,-7,-3,-5,-2,-3,-4, 0,-6,-2,17, 0,-6,-5,-3,-6,-1,-8}
    ,   {-3, 0,-3,-4,-4,-2,-4,-4,-5, 0,-1,-1,-4,-2, 7,-5,-3, 0,10,-2,-3,-1,-4,-1,-8}
    ,   { 0,-2, 0,-2,-2,-2,-2,-2,-1,-2, 4, 2,-2, 2,-1,-1,-1,-6,-2, 4,-2, 2,-2,-1,-8}
    ,   { 0,-4, 0, 3,-1, 2, 3, 1, 0, 1,-2,-3, 1,-2,-4,-1, 0,-5,-3,-2, 3,-3, 2,-1,-8}
    ,   {-1,-5,-1,-3,-3,-3,-3,-2,-4,-2, 3, 5,-3, 3, 2,-2,-2,-3,-1, 2,-3, 5,-2,-1,-8}
    ,   { 0,-5,-1, 3, 0, 1, 3, 3, 0, 2,-2,-3, 0,-2,-5, 0, 0,-6,-4,-2, 2,-2, 3,-1,-8}
    ,   {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-8}
    ,   {-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8, 0}
    }
};

/*
 * Maps the table string name to its respective table index. This will be needed
 * to translate the table name from string to its integer.
 */
static std::map<std::string, int> tablemap = {
    {tablenames[0], 0}
,   {tablenames[1], 1}
,   {tablenames[2], 2}
,   {tablenames[3], 3}
,   {tablenames[4], 4}
,   {tablenames[5], 5}
};

/*
 * Aliases a table line to a single symbol. This allows an easier memory
 * allocation and usage for the scoring table.
 * @since 0.1.alpha
 */
using Line = int8_t[25];

/** 
 * Loads a scoring table into the device. The table will be automatically freed
 * when algorithm is destructed.
 */
void pairwise::Algorithm::loadBlosum()
{
    int index = tablemap[cli.get("matrix")];

    onlyslaves {
        Line *table;
        
        device::malloc(table, sizeof(Line) * 25);
        device::memcpy(table, &tabledata[index], sizeof(Line) * 25);

        this->table = {table, device::free<Line>};
        this->penalty = abs(tabledata[index][24][0]);
    }

    onlymaster info("using scoring table %s", tablenames[index].c_str());
}
