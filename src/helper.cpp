/** 
 * Multiple Sequence Alignment helper functions file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <stdexcept>
#include <cstdio>

#include "config.h"
#include "helper.h"

#ifndef msa_disable_cluster
#include "node.hpp"
#include "cluster.hpp"
#endif

/**
 * Prints out a message of help for the user. The message uses the description
 * of options given and set on main. The message consists of all command line
 * options currently available. The name, arguments and a brief description of
 * each command are also shown.
 * @see main
 */
void usage()
{
#ifndef msa_disable_cluster
    onlymaster {
        fprintf(stderr, s_bold "[  usage]:" s_reset " msarun [options] filename\n");
    }

    finalize({});
#endif
}

/**
 * Prints out the software's current version. This is important so the user can
 * know whether they really are using the software they want to. The version of
 * this software is defined in the main header file, and should not be hardwired.
 * @see inc/msa.hpp
 */
void version()
{
#ifndef msa_disable_cluster
    onlymaster {
        fprintf(stderr, s_bold "[version]:" s_reset " %s\n", msa_version);
    }

    finalize({});
#endif
}

/**
 * Aborts the execution and indicates the execution shuold terminate. In case this
 * function is being compiled for test purposes, in throws an exception.
 * @param error Error detected during execution.
 */
void finalize(Error error)
{
#ifndef msa_disable_cluster
    if(error.msg != NULL) {
        fprintf(stderr, s_bold "[  error]: " c_red_fg "fatal" s_reset ": %s\n", error.msg);
    }

    cluster::finalize();
    exit(0);
#else
    throw std::logic_error(error.msg);
#endif
}