#!/usr/bin/env bash
# Multiple Sequence Alignment cluster watchdog script.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018 Rodrigo Siqueira

# The watchdog script requires as the first positional command line option, the PID
# of the process it is watching. Also, the output of the target process must be
# redirected to this script's stdin.

# Declaring escape sequences and style constants.
readonly s_bold="\033[1m"
readonly s_reset="\033[0m"
readonly c_red_fg="\033[31m"
readonly c_blue_fg="\033[34m"
readonly c_green_fg="\033[32m"
readonly e_clearline="\033[2K"

# Declaring the spinner frames.
declare -a spinner=("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏")

# Checking whether the PID has been sent to watcher. If the PID is not known, the
# script must exit as an error.
if [ $# -lt 1 ]; then
    echo "[error] watchdog requires the PID of process it watches."
    exit 1
fi

# Initializing the progress bar variables.
bar_counter=0
bar_percent=0
bar_message=""
bar_process="idle"

# Initializing the percent tracking variables.
elements_done=()
elements_total=()

# Catch the PID of process to watch.
readonly target_pid=$1

# Updates the progress percentage according to parameters sent to the function.
# The percentage calculated will be automatically updated in the progress bar.
update()
{
    # Declaring names for function parameters.
    local cluster_node=$1
    local cluster_total=$2
    local node_done=$3
    local node_total=$4

    # Updating node values according to parameters.
    elements_done[$cluster_node]=$node_done
    elements_total[$cluster_node]=$node_total

    # Calculating the total of completed tasks.
    local tdone=$(echo "${elements_done[@]/%/+}0" | bc)
    local total=$(echo "${elements_total[@]/%/+}0" | bc)

    # Updating the progress bar.
    [ $total -gt 0 ] && bar_percent=$((tdone * 100 / total)) || bar_percent=0
}

# Parses a line read from the target process. The line may request a progress update
# or a progress reset, report an error or simply be a line to be printed.
parseline()
{
    # The line contents to be parsed.
    local line_contents=($@)

    # If the line has been targeted to the watchdog, then update the progress.
    if [ "${line_contents[0]}" == "[watchdog]" ]; then
        bar_process="${line_contents[1]}"
        bar_message="${line_contents[@]:6}"
        update $3 $4 $5 $6

    # Reset the watchdog if the process request so.
    elif [ "${line_contents[0]}" == "[watchdog.clean]" ]; then
        bar_message=""
        bar_process="idle"
        elements_done=()
        elements_total=()
        update 0 0 0 0

    # If the process has reported an error, then it shall be killed.
    elif [ "${line_contents[0]}" == "[error]" ]; then
        kill -9 $target_pid > /dev/null 2>&1
        error "${line_contents[@]:1}"
        exit 1

    # If the line is not targeted to the watchdog, simply print it.
    else
        printf "${e_clearline}\r   %s\n" \
            "${line_contents[*]}"
    fi
}

# Prints the progress bar based on the bar control variables. The progress bar is
# responsible for keeping a spinner on screen while the process is active.
progressbar()
{
    printf "${e_clearline}\r %s ${c_green_fg}${s_bold}[%3d%%] ${c_blue_fg}%s${s_reset} %s" \
        "${spinner[$bar_counter]}" $bar_percent $bar_process "$bar_message"
    bar_counter=$(((bar_counter + 1) % ${#spinner[@]}))
}

# Reports an error for the user. This function allows a message to be sent and it
# will be shown on the progress bar before killing the target process.
error()
{
    local args=($@)
    printf "${e_clearline}\r   ${c_red_fg}${s_bold}[error] ${c_blue_fg}%s${s_reset} %s\n" \
        $bar_process "${args[*]}"
}

# Analyze all input coming in the standard input channel while the target process
# is active. Keep printing progress bar while process is calculating.
while kill -0 "$target_pid" > /dev/null 2>&1; do
    # Reading line from standard input.
    IFS= read -r -t .1 line <&0

    # Check if a line has been read. If yes, parse the line and if any error has
    # been detected, then exit the script.
    if [[ ! -z "$line" ]] && ! parseline $line; then
        exit 1
    fi

    # Prints the updated progress bar.
    progressbar
done

# When the target process has already been terminated, all remaining lines printed
# by the process before exiting must be analyzed.
while IFS= read -r line; do
    # Parse the line read and if any error has been detected, then exit.
    if ! parseline $line; then
        exit 1
    fi

    # Prints the updated progress bar.
    progressbar
done <&0

# Cleaning the progress bar spinner.
printf "\r  \n"
