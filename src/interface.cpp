/** @file interface.cpp
 * @brief Parallel Multiple Sequence Alignment command line interface file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <iostream>
#include <iomanip>
#include <cstring>

#include "msa.h"
#include "interface.hpp"

struct cli_data cli_data;

using namespace std;

/** @fn void help(int, char **, int*)
 * @brief Prints out the help menu.
 * @param pname The name used to start the software.
 */
void help(int, char **argv, int *)
{
    cerr << "Usage: "  << argv[0] << " [options] --file fname" << endl;
    cerr << "Options:" << endl;

    for(int i = 0; cli_command[i].abbrev; ++i)
        cerr << setw(4) << right << cli_command[i].abbrev   << ", "
             << setw(9) << left  << cli_command[i].complete
             << setw(9) << left  << (cli_command[i].arg ? " arg" : " ")
             << cli_command[i].desc << endl;

    finish(0);
}

void version(int, char **, int *)
{
    cerr << setw(4) << left << MSA << VERSION << endl;
    finish(0);
}

void file(int argc, char **argv, int *j)
{
    if(argc <= *j + 1)
        finish(0);

    cli_data.fname = argv[++*j];
}

/** @fn void unknown(int, char **, int *)
 * @brief Informs the user an unknown argument was used.
 * @param arg The unknown argument used.
 */
void unknown(int, char **argv, int *j)
{
    cerr << "Unknown option: " << argv[*j] << endl;
    cerr << "Try `" << argv[0] << " -h' for more information." << endl;
    finish(0);
}

struct cli_command cli_command[] = {
    {&help,    "-h", "--help",    "Displays this help menu."}
,   {&version, "-v", "--version", "Displays the version information."}
,   {&file,    "-f", "--file",    "File to be loaded into application.", true}
,   {&unknown}
};

struct cli_command *search(const char *arg)
{
    int i;

    for(i = 0; cli_command[i].abbrev || cli_command[i].complete; ++i)
        if(!strcmp(cli_command[i].abbrev, arg) || !strcmp(cli_command[i].complete, arg))
            break;

    return &cli_command[i];
}

/** @fn void parse_cli(int, char **)
 * @brief Parses the command line arguments and fills up the command line struct.
 * @param argc Number of arguments sent by command line.
 * @param argv The arguments sent by command line.
 */
void parse_cli(int argc, char **argv)
{
    struct cli_command *actual;

    for(int i = 1; i < argc; ++i) {
        actual = search(argv[i]);
        (actual->func)(argc, argv, &i);
    }
}
