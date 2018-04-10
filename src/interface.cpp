/*! \file interface.cpp
 * \brief Parallel Multiple Sequence Alignment command line interface file.
 * \author Rodrigo Siqueira <rodriados@gmail.com>
 * \copyright 2018 Rodrigo Siqueira
 */
#include <iostream>
#include <getopt.h>

#include "msa.hpp"
#include "interface.hpp"

using namespace std;

/*! \fn help(char *)
 * Prints out the help menu.
 * \param pname The name used to start the software.
 */
void help(char *pname)
{
    cerr << "Usage: " << pname << " [options] file" << endl;
    cerr << "Options:" << endl;
    cerr << "  -f file     File to be loaded into application" << endl;
    cerr << "  -h          Displays this help menu" << endl;
    cerr << endl;
    cerr << "version: " << VERSION << endl;
}

/*! \fn unknown(char *)
 * Informs the user an unknown argument was used.
 * \param arg The unknown argument used.
 */
void unknown(char arg)
{
    cerr << "Unknown option: -" << arg << endl;
    cerr << "Try `-h' for more information" << endl;
}

/*! \fn argparse(int, char **)
 * Parses the command line arguments and fills up the command line struct.
 * \param argc Number of arguments sent by command line.
 * \param argv The arguments sent by command line.
 */
void argparse(int argc, char **argv)
{
    char option;

    if(argc < 2) {
        help(argv[0]);
        finish();
    }

    while((option = getopt(argc, argv, "hf:")) != -1)
        switch(option) {
            case 'h':
                help(argv[0]);
                finish();

            case 'f':
                gldata.fname = optarg;
                break;

            case '?':
                unknown(optopt);

            default:
                finish();
        }

    gldata.fname = gldata.fname
        ? gldata.fname
        : argv[optind];
}
