/**
 * Multiple Sequence Alignment pairwise scoring tables file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#include <map>
#include <string>
#include <cstdint>

#include <msa.hpp>
#include <cuda.cuh>
#include <pointer.hpp>
#include <exception.hpp>

#include <pairwise/pairwise.cuh>

namespace msa
{
    /**
     * Aliasing the scoring table's raw type in order to avoid excessive verbosity
     * with the fully-qualified type name.
     * @since 0.1.1
     */
    using table_type = pairwise::scoring_table::raw_type;

    /**
     * Aliasing the scoring table's raw element type in order to avoid excessive
     * verbosity with the fully-qualified type name.
     * @since 0.1.1
     */
    using element_type = pairwise::scoring_table::element_type;

    /**
     * The raw scoring tables data. One of these tables will be transfered to device
     * memory so it can be used to score sequence alignments. The first table,
     * index-zero, is used as the default, when no valid parameter is found to
     * indicate which table should be used instead.
     * @since 0.1.1
     */
    static table_type table_data[] = {
        {   /* [0] default */
            /*A  C  T  G  R  N  D  Q  E  H  I  L  K  M  F  P  S  W  Y  V  B  J  Z  X  **/
            { 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
        ,   {-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
        ,   {-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
        ,   {-1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
        ,   {-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
        ,   {-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
        ,   {-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
        ,   {-1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
        ,   {-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
        ,   {-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
        ,   {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
        ,   {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
        ,   {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
        ,   {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
        ,   {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
        ,   {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
        ,   {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1}
        ,   {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1}
        ,   {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1}
        ,   {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1}
        ,   {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1}
        ,   {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1}
        ,   {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1}
        ,   {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
        ,   {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0}
        }
    ,   {   /* [1] blosum62 */
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
    ,   {   /* [2] blosum45 */
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
    ,   {   /* [3] blosum50 */
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
    ,   {   /* [4] blosum80 */
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
    ,   {   /* [5] blosum90 */
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
    ,   {   /* [6] pam250 */
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

    /**
     * The list of all available scoring tables' names.
     * @since 0.1.1
     */
    static const std::vector<std::string> table_name = {
        "default"
    ,   "blosum62"
    ,   "blosum45"
    ,   "blosum50"
    ,   "blosum80"
    ,   "blosum90"
    ,   "pam250"
    };

    /**
     * Represents a local and simple scoring table instance. This struct is used
     * to link the static tables to their respective penalty values.
     * @since 0.1.1
     */
    typedef struct {
        table_type * const data;
        const element_type penalty;
    } local_table;

    /**
     * Maps the table's string names to their respective info. This will be needed
     * when translating a table's names from a string to its actual contents.
     */
    static const std::map<std::string, const local_table> dispatcher = {
        {table_name[0], {&table_data[0], 1}}
    ,   {table_name[1], {&table_data[1], 4}}
    ,   {table_name[2], {&table_data[2], 5}}
    ,   {table_name[3], {&table_data[3], 5}}
    ,   {table_name[4], {&table_data[4], 6}}
    ,   {table_name[5], {&table_data[5], 6}}
    ,   {table_name[6], {&table_data[6], 8}}
    };

    /**
     * Transfers the selected scoring table into device memory.
     * @return The new scoring table instance.
     */
    auto pairwise::scoring_table::to_device() const -> pairwise::scoring_table
    {
        auto ptr = pointer<table_type>::make(cuda::allocator::device);
        cuda::memory::copy(&ptr, &m_contents);
        
        return {ptr, m_penalty};
    }

    /**
     * Gets the description of a scoring table selected by name.
     * @param name The name of selected scoring table.
     * @return The local description of requested table.
     */
    static const local_table& retrieve(const std::string& name)
    {
        const auto& pair = dispatcher.find(name);

        enforce(pair != dispatcher.end(), "could not find scoring table <bold>%s</>", name);
        onlymaster watchdog::info("selected pairwise scoring table <bold>%s</>", name);

        return pair->second;
    }

    /**
     * Selects a scoring table from its name.
     * @param name The name of selected scoring table.
     * @return The pointer to selected table.
     */
    auto pairwise::scoring_table::make(const std::string& name) -> pairwise::scoring_table
    {
        const local_table& selected = retrieve(name);
        return {pointer<table_type>::weak(selected.data), selected.penalty};
    }

    /**
     * Informs the names of all available scoring tables.
     * @return The list of available scoring tables.
     */
    auto pairwise::scoring_table::list() noexcept -> const std::vector<std::string>&
    {
        return table_name;
    }
}