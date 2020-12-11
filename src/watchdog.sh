#!/usr/bin/env bash
# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file The script for the execution watchdog routine.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-present Rodrigo Siqueira

# This script watches over the output of a process and exhibits the process's progress.
# To interact with the watchdog, a process must use the format given in this file,
# more specifically, the same pattern as the one shown in the function 'wnotify'.
# @param $1 The PID of process to watch over.

# Including helping script functions.
source src/utils.sh

# Runs a watchdog process.
# @param $1 The PID of process to watch.
watchdog()
{
    local pid=0
    local reportonly=

    # Parses the command line options and sets all respective variables. Any positional
    # values not linked to any option sent will be treated as PID.
    while [ $# -gt 0 ]; do
        case $1 in
            -r | --report-only  ) reportonly="$1";  shift ;;
            *                   ) pid="$1";         shift ;;
        esac
    done

    # Kills the target process with given status number.
    # @param $1 The status number to the quit the script with.
    prockill()
    {
        kill -2 -$pid > /dev/null 2>&1;     sleep 1
        kill -9  $pid > /dev/null 2>&1
        exit $1
    }

    # Formats and prints an info message to the standard output.
    # @param $1 The info message to be printed.
    info()
    {
        if [[ -z "$reportonly" ]]; then
            printf "\r%27s · %s\n" "$(style clean bold blue info)" "$(markdown "$1")"
        fi
    }

    # Formats and prints an error message to the standard output.
    # @param $1 The error message to be printed.
    error()
    {
        if [[ -z "$reportonly" ]]; then
            printf "\r%27s · %s\n" "$(style clean bold red error)" "$(markdown "$1")"
            prockill 1
        else
            printf "error: %s\n" "$1"
            prockill 1
        fi
    }

    # Formats and prints a warning message to the standard output.
    # @param $1 The warning message to be printed.
    warning()
    {
        if [[ -z "$reportonly" ]]; then
            printf "\r%27s · %s\n" "$(style clean bold yellow warning)" "$(markdown "$1")"
        else
            printf "warning: %s\n" "$1"
        fi
    }

    # Aborts the script execution and imediately halts the watched process execution.
    # First, a SIGINT will be sent to process, and then SIGKILL if still alive.
    abort()
    {
        if [[ -z "$reportonly" ]]; then
            printf "\r%27s · abort manually requested by user\n" "$(style clean bold red abort)"
            prockill 2
        else
            echo "abort manually requested by user"
            prockill 2
        fi
    }

    # Initializing the current process' informative variables.
    local process_name="idle"
    local process_message=""
    local process_active=0
    local blink_state=0

    # Initializes the watchdog for keeping track of progress of a new process.
    # @param $1 The name of process to watch.
    # @param $2 The description of process to watch.
    init()
    {
        if [[ -z "$reportonly" ]]; then
            process_name="$1"
            process_message="$(markdown "$2")"
            process_active=1
        fi
    }

    # Notifies the conclusion of the watched process.
    # @param $1 The name of process being watched and which has just finished.
    # @param $2 The process' conclusion message.
    finish()
    {
        # We check whether the operation corresponds to the currently watched
        # process, just to be sure no parallel creepyness is going on.
        if [[ -z "$reportonly" && "$1" == "$process_name" ]]; then
            printf "\r%27s · %12s · %s\n" "$(style clean bold green "$process_name")"       \
                "$(style bold "done")"                                                      \
                "$(markdown "$2")"
            process_active=0
        fi
    }

    # Reports the time spent by a step of the watched process to process.
    # @param $1 The process name to report execution time.
    # @param $2 The time spent by the process executing.
    report()
    {
        if [[ -z "$reportonly" ]]; then
            printf "\r%27s · completed in %s seconds\n" "$(style clean bold green "$1")"    \
                "$(style bold "$2")"
        else
            printf "report: %s: $2 seconds\n" "$1"
        fi
    }

    # The unparsed input cache. This is necessary due to the unpredictability of
    # a parallel program execution, and thus, lines may be printed inside of the
    # following totally unrelated line.
    local unparsed_input=""        

    # Parses a line read from the target process. The line may request a progress
    # update or a progress reset, report an error or simply be a line to print.
    # @param $1 The line to be parsed and analyzed.
    parse()
    {
        # The current content to parse. We concatenated any previously unparsed
        # contents so it may be useful for the current line.
        local contents="${unparsed_input}${1}"

        # Check whether the current input is a watchdog line and captures its info.
        if [[ $contents =~ ^(.*)\[watchdog\|(info|error|warning|init|finish|report)\|(.*)\]$ ]]; then
            # Save the unparsed contents so it may be useful for the next line.
            unparsed_input="${BASH_REMATCH[1]}"

            # Captures the action parsed by the regular expression.
            local action=${BASH_REMATCH[2]}

            # Splits the input and allows individual values to be accessed.
            IFS=\| read -a fields <<< "${BASH_REMATCH[3]}"

            # Executes the requested watchdog action.
            $action "${fields[@]}"

        # If the pattern is not matched, then we shall simply print everything.
        else
            # Resets the unparsed contents as everything will be printed.
            unparsed_input=""

            # Prints out the pure line contents.
            echo "$(style clean "$contents")"
        fi
    }

    # Catch a user Ctrl-C and perform abort routine.
    trap "abort" 2

    # Prints out the watchdog's process' progress information.
    progress()
    {
        # Checks whether there is any active process at the moment.
        if [ $process_active -eq 1 ]; then
            printf "\r%27s · %12s %s %s"                                                    \
                "$(style clean bold green "$process_name")"                                 \
                "$(style bold "running")"                                                   \
                "$([ $blink_state -eq 0 ] && echo "·" || echo " ")"                         \
                "$process_message"
            blink_state=$((blink_state > 0 ? 0 : 1))
        fi
    }

    # Analyze all input coming from the standard input channel while the target
    # process is active. Keep printing progress bar while process is calculating.
    while kill -0 "$pid" > /dev/null 2>&1; do
        # Reading the target process output.
        IFS= read -r -t .5 line <&0

        # Checks whether a line has been read. If so, then parse the line and
        # execute it's requested action.
        if [[ ! -z "$line" ]] && ! parse "$line"; then
            exit 1
        fi

        # Reprint the watchdog's progress bar.
        progress
    done

    # When the target process has already been terminated, all remaining lines
    # printed by the process before exiting must be analyzed.
    while IFS= read -r line; do
        # Parse the remaining lines and checks if any error has been detected.
        # If an error happens to be detected, then exit the script.
        if ! parse "$line"; then
            exit 1
        fi
    done <&0
}

# Creates a watchdog notification with given parameters.
# @param $1 The notification type.
# @param $@ The notification's parameters values.
# @return The newly created watchdog notification.
wnotify()
{
    # The variable containing the nofitication's constructed string. This is will
    # be the function's final and returned result.
    local notification="[watchdog"$(printf "|%s" "${@:1}")"]"

    # Returning the built notification.
    printf "$notification\n"
}

# Checks whether this file is being sourced. If it is being sourced, then we should
# not execute any action as this might be undesirable.
if [ "$0" = "$BASH_SOURCE" ]; then
    # Execute the watchdog process on the given PID.
    watchdog $@ <&0
fi
