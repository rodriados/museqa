#!/usr/bin/env bash
# Multiple Sequence Alignment utility functions script file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2019 Rodrigo Siqueira

# Applies a new style to given text.
# @param $@ List of style names to apply
# @param $-1 The text to apply style to.
# @return The stylized text.
style()
{
    # Creates an empty styling string. This string will be concatenated with all
    # the styles requested to the function.
    local applied_style=""

    # The formatting reset flag. This flag is used at the end of every styled string.
    local reset="\033[0m"

    # Iterates over all of the function's parameters except the last one. For each
    # parameter, a new style configuration will be concatenated.
    while [ $# -gt 1 ]; do
        case $1 in
            bold        ) applied_style="${applied_style}\033[1m";  shift ;;
            dim         ) applied_style="${applied_style}\033[2m";  shift ;;
            italic      ) applied_style="${applied_style}\033[3m";  shift ;;
            underline   ) applied_style="${applied_style}\033[4m";  shift ;;
            red         ) applied_style="${applied_style}\033[31m"; shift ;;
            green       ) applied_style="${applied_style}\033[32m"; shift ;;
            yellow      ) applied_style="${applied_style}\033[33m"; shift ;;
            blue        ) applied_style="${applied_style}\033[34m"; shift ;;
            normal      ) applied_style="${applied_style}\033[39m"; shift ;;
            clean       ) applied_style="${applied_style}\033[2K";  shift ;;
            *           )                                           shift ;;
        esac
    done

    # Returns the stylized string.
    printf "$applied_style$1$reset"
}

# Analyzes a given string and applies styles to it as requested by its formatting.
# This works as if it was a very simple markdown parser, allowing the same commands
# as those of the 'style' function.
# @param $1 The string to be parsed.
# @return The stylized text.
markdown()
{
    # Catches the original string to be parsed.
    local target="$1"

    # Checks whether there still are styleable commands to parse.
    while [[ $target =~ ^(.*)\<([ a-z]+)\>(.*)\</\>(.*)$ ]]; do
        # Applies the requested styling to part of the target string.
        local stylized="$(style ${BASH_REMATCH[2]} "${BASH_REMATCH[3]}")"

        # Rebuilds the target string with the new stylized content.
        target="${BASH_REMATCH[1]}${stylized}${BASH_REMATCH[4]}"
    done

    # Returns the final stylized string.
    printf "$target"
}

# Creates a new named pipe and returns its name. The pipe allows the establishment
# of communication to and from an external process.
# @return The name of created pipe.
makepipe()
{
    # Creates a named pipe. The name of the pipe will simply be that of a temporary
    # file, which existence is guaranteed while the pipe is open.
    local pipe=$(mktemp -u)
    mkfifo $pipe

    # Return the name of pipe created.
    echo $pipe
}
