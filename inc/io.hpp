/**
 * Multiple Sequence Alignment IO module interface header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2020 Rodrigo Siqueira
 */
#pragma once

#include <io/io.hpp>
#include <io/loader/database.hpp>

namespace msa
{
    /**
     * Exposes the command line manager object to the root namespace.
     * @since 0.1.1
     */
    typedef io::cmdline cmdline;
}
