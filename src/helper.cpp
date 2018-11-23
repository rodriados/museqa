/** 
 * Multiple Sequence Alignment helper functions file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <cstdarg>
#include <cstdio>

#include "helper.hpp"
#include "cluster.hpp"

/**
 * Prints information about a error caught and quits the execution.
 * @param format The formating string to print.
 */
[[noreturn]] void error(const char *format, ...)
{
    va_list args;
    va_start(args, format);

    printf("[error] ");
    vprintf(format, args);
    putchar('\n');

    va_end(args);

    fflush(stdout);
    cluster::finalize();
    exit(1);
}

/**
 * Shows the current software version. This macro function serves as the current
 * software version to all modules.
 */
void version()
{
    printf("[version] " s_bold c_green_fg msa_version s_reset "\n");
    fflush(stdout);
}
