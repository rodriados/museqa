/** 
 * Multiple Sequence Alignment interface header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _INTERFACE_HPP_
#define _INTERFACE_HPP_

#include <cstdint>

namespace cli {

/** 
 * Lists all possible command codes from the terminal.
 * @since 0.1.alpha
 */
enum class Code : uint8_t
{
    CLI_UNKN = 0
,   CLI_HELP
,   CLI_VERS
,   CLI_VERB
,   CLI_FILE
,   CLI_MGPU
,   CLI_MTRX
};

/** 
 * Holds strings needed for creating command line options and arguments.
 * @since 0.1.alpha
 */
struct Command final
{
    const Code id;           /// The command code.
    const char *abb = NULL;     /// The abbreviated option tag.
    const char *full = NULL;    /// The full option tag.
    const char *desc = NULL;    /// The option description.
    const char *arg = "";       /// If an argument is needed, what is its name?
};


/**
 * Holds obtained via command line parsing.
 * @since 0.1.alpha
 */
struct Data final
{
    const char *fname = NULL;   /// The name of file to be processed.
    const char *matrix = NULL;  /// The scoring matrix to use.
    bool multigpu = false;      /// Indicates whether to use multiple devices.
};

extern void parse(int, char **);

}

#endif