/** @file interface.hpp
 * @brief Parallel Multiple Sequence Alignment interface header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _INTERFACE_HPP
#define _INTERFACE_HPP

/** @enum command
 * @brief Lists all possible commands from the terminal.
 */
enum command : short int {
    CLI_UNKN = 0,
    CLI_HELP,
    CLI_VERS,
    CLI_VERB,
    CLI_FILE,
};

/** @struct cli_command
 * @brief Holds strings needed for creating command line options and arguments.
 * @var abb The abbreviated option name.
 * @var full The full option name.
 * @var desc The option or argument description.
 * @var arg If an argument is needed, what is its name?
 */
struct cli_command {
    const enum command id;
    const char *abb = NULL;
    const char *full = NULL;
    const char *desc = NULL;
    const char *arg = "";
};


/** @struct cli_data
 * @brief Holds obtained via command line parsing.
 * @var fname The name of file to be processed.
 */
struct cli_data {
    const char *fname = NULL;
};

extern struct cli_data cli_data;
extern void parsecli(int, char **);

#endif