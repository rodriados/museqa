/** @file interface.hpp
 * @brief Parallel Multiple Sequence Alignment interface header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _INTERFACE_HPP
#define _INTERFACE_HPP

#include <cstdint>

/** @enum command_t
 * @brief Lists all possible commands from the terminal.
 */
typedef enum : uint8_t {
    CLI_UNKN = 0
,   CLI_HELP
,   CLI_VERS
,   CLI_VERB
,   CLI_FILE
,   CLI_MGPU
,   CLI_MTRX
} command_t;

/** @struct clicommand_t
 * @brief Holds strings needed for creating command line options and arguments.
 * @var abb The abbreviated option name.
 * @var full The full option name.
 * @var desc The option or argument description.
 * @var arg If an argument is needed, what is its name?
 */
typedef struct {
    const command_t id;
    const char *abb = NULL;
    const char *full = NULL;
    const char *desc = NULL;
    const char *arg = "";
}  clicommand_t;


/** @struct clidata_t
 * @brief Holds obtained via command line parsing.
 * @var fname The name of file to be processed.
 * @var matrix The scoring matrix to use.
 * @var multigpu Indicates whether multiple devices should be used.
 */
typedef struct {
    const char *fname = NULL;
    const char *matrix = NULL;
    bool multigpu = false;
} clidata_t;

namespace cli
{
    extern void parse(int, char **);
}

#endif