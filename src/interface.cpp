/** @file interface.cpp
 * @brief Parallel Multiple Sequence Alignment command line interface file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cstring>

#include "msa.hpp"
#include "interface.hpp"

cli::Data cli_data;
cli::Command cli_command[] = {
    {CLI_HELP, "-h", "--help",     "Displays this help menu."}
,   {CLI_VERS, "-e", "--version",  "Displays the version information."}
,   {CLI_VERB, "-v", "--verbose",  "Activates the verbose mode."}
,   {CLI_FILE, "-f", "--file",     "File to be loaded into application.", "fn"}
,   {CLI_MGPU, "-m", "--multigpu", "Use multiple GPU devices if possible."}
,   {CLI_MTRX, "-x", "--matrix",   "Inform the scoring matrix to use.", "mat"}
,   {CLI_UNKN}
};

namespace cli {

/** 
 * Gets the name of file to be loaded and processed.
 * @param argc The number of command line arguments.
 * @param argv The command line arguments.
 * @param i The current command line argument index.
 */
void file(int argc, char **argv, int *i)
{
    if(argc <= *i + 1)
        finalize(NOFILE);

    cli_data.fname = argv[++*i];
}

/** 
 * Informs the scoring matrix to be used.
 * @param argc The number of command line arguments.
 * @param argv The command line arguments.
 * @param i The current command line argument index.
 */
void matrix(int argc, char **argv, int *i)
{
    if(argc <= *i + 1)
        finalize(INVALIDARG);

    cli_data.matrix = argv[++*i];
}

/**
 * Prints out the help menu.
 * @param pname The name used to start the software.
 */
void help(char *pname)
{
    std::stringstream ss;

    std::cerr << "Usage: "  << pname << " [options] -f fn" << std::endl;
    std::cerr << "Options:" << std::endl;

    for(int i = 0; cli_command[i].abb; ++i) {
        ss << cli_command[i].full << ' ' << cli_command[i].arg;
        std::cerr << std::setw(4)  << std::right << cli_command[i].abb << ", "
                  << std::setw(15) << std::left  << ss.str()
                  << cli_command[i].desc << std::endl;
        ss.str(std::string());
    }

    finalize(NOERROR);
}

/**
 * Prints out the software version number.
 */
void version()
{
    std::cerr << std::setw(4) << std::left << MSA << VERSION << std::endl;
    finalize(NOERROR);
}

/** 
 * Informs the user an unknown argument was used.
 * @param pname The name used to start the software.
 * @param cm The unknown command detected.
 */
void unknown(char *pname, char *cm)
{
    std::cerr << "Unknown option: " << cm << std::endl;
    std::cerr << "Try `" << pname << " -h' for more information." << std::endl;
    finalize(NOERROR);
}

/**
 * Searches for a command in the command list.
 * @param cm Command to be searched for.
 * @return The selected command.
 */
Command *search(const char *cm)
{
    int i;

    for(i = 0; cli_command[i].abb || cli_command[i].full; ++i)
        if(!strcmp(cli_command[i].abb, cm) || !strcmp(cli_command[i].full, cm))
            break;

    return &cli_command[i];
}

}

/** 
 * Parses the command line arguments and fills up the command line struct.
 * @param argc Number of arguments sent by command line.
 * @param argv The arguments sent by command line.
 */
void cli::parse(int argc, char **argv)
{
    for(int i = 1; i < argc; ++i)
        switch(search(argv[i])->id) {
            case CLI_VERB: verbose = 1;                     break;
            case CLI_VERS: version();                       break;
            case CLI_HELP: help(argv[0]);                   break;
            case CLI_FILE: file(argc, argv, &i);            break;
            case CLI_MGPU: cli_data.multigpu = true;        break;
            case CLI_MTRX: matrix(argc, argv, &i);          break;
            case CLI_UNKN: unknown(argv[0], argv[i]);       break;
        }

    if(!cli_data.fname)
        finalize(NOFILE);
}