/** 
 * Multiple Sequence Alignment helper functions file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <stdexcept>
#include <cstdio>
#include <map>

#include "config.h"
#include "helper.hpp"

#ifndef msa_disable_cluster
#include "cluster.hpp"
#endif

/*
 * Maps error severity to its name. This is useful so we can show the user what
 * kind of error was thrown and detected in the code.
 */
static const std::map<int, const char *> errname = {
    {ErrorSuccess,                ""}
,   {ErrorWarning,                "warning"}
,   {ErrorRuntime,                "runtime"}
,   {ErrorFatal,                  "fatal"}
};

/**
 * Aborts the execution and indicates the execution shuold terminate. In case this
 * function is being compiled for test purposes, in throws an exception.
 * @param error Error detected during execution.
 */
void finalize(Error error)
{
#ifndef msa_disable_cluster
    errlog(error);
    cluster::finalize();
    exit(0);
#else
    throw std::logic_error(error.msg);
#endif
}

/**
 * Prints information about a caught error or warning.
 * @param error The error to be reported.
 */
void errlog(Error error)
{
    if(error.severity & ErrorWarning) {
        printf("[warning] %s\n", error.msg.c_str());
        fflush(stdout);
    }

    else if(error.severity != ErrorSuccess) {
        printf("[error][%s] %s\n", errname.at(error.severity), error.msg.c_str());
        fflush(stdout);
    }
}

/**
 * Reports the progress of a given task.
 * @param taskname The name of the task to be reported.
 * @param done The amount of the task that is already done.
 * @param total The total to be processed by the task.
 */
void progress(const char *taskname, uint32_t done, uint32_t total)
{
    printf("[progress] %s %u %u\n", taskname, done, total);
    fflush(stdout);
}
