/** @file interface.hpp
 * @brief Parallel Multiple Sequence Alignment interface header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _INTERFACE_HPP
#define _INTERFACE_HPP

/** @struct cli_command
 * @brief Holds strings needed for creating command line options and arguments.
 * @var abbrev The abbreviated option name.
 * @var complete The full option name.
 * @var desc The option or argument description.
 * @var arg Is there a variable expecting to get the arguments value?
 */
struct cli_command {
    void (*func)(int, char **, int *);
    const char *abbrev = NULL;
    const char *complete = NULL;
    const char *desc = NULL;
    const bool arg = false;
};

struct cli_data {
    const char *fname;
};

extern struct cli_data cli_data;
extern struct cli_command cli_command[];

extern void parse_cli(int, char **);

#endif