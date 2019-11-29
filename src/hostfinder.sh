#!/usr/bin/env bash
# Multiple Sequence Alignment hosts finder script.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2019 Rodrigo Siqueira

# This script traverses the current local network in search of all computer-capable
# devices we can run our software on, and automatically generates a hostfile for
# MPI execution. This search is done by accessing every host individually via SSH
# and asking how many available devices each one of them has. In case the user desires
# a finer control of the nodes, or even use external nodes, the file may be created
# manually before the software's execution.
# @param $1 The name of hostfile to be created.

# Checks whether this file is being sourced. If it is being sourced, then we should
# not execute any action as this is undesirable.
if [ "$0" = "$BASH_SOURCE" ]; then
    # Including watchdog notification functions.
    source src/watchdog.sh

    # Command to check whether a host is compute capable or not.
    readonly gpu_counter="lspci | grep 'VGA' | grep 'NVIDIA' | wc -l"

    # Checks whether a host has a compute capable device. This is done by simply connecting
    # to the host and executing a command to check whether it is compute capable or not.
    # @param $1 The target host to count GPUs from.
    # @return The number of GPUs found in host.
    sshrun()
    {
        # Access remote host in batch mode without tty allocation and runs the command
        # to find out the number of connected GPUs.
        ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no -T $1 << EOF
            ${gpu_counter}
            exit
EOF
    }

    # Automatically generates a hostfile for MPI execution.
    # @param $1 The name of hostfile to be created
    generate()
    {
        wnotify init "hostfinder" "finding available devices in current network"

        # Discovers how many slots we can give to the local host.
        local gpu_local=$(eval "$gpu_counter")
        local gpu_total=$gpu_local
        
        # Checks whether this machine is CUDA compute capable and checks how many devices
        # are currently available.
        echo "localhost slots=$((gpu_local + 1)) max-slots=$((gpu_local + 1))" >> "$1"

        # Gets the list of hosts known by the current machine in local network.
        local host_list=$(getent hosts | cut -d' ' -f1 | tr "\n" " ")
        local host_count=$(wc -w <<< "$host_list")

        # Discovers the current network IP address of local machine.
        local ip_local=$(hostname -I)

        # Reports the current execution progress
        local loop_count=0

        # Iterating through the list of hosts to try to find hosts capable of running
        # the software. As MPI uses ssh internally to implement communication between
        # nodes, we use ssh here as well to look for 'em hosts.
        for host in ${host_list[@]}; do
            # Notifies the watchdog about the execution's progress.
            wnotify update "hostfinder" 0 $loop_count $host_count

            # Checking for local hosts only. We do not want to use external nodes when
            # executing. The user shall create the hosts file manually otherwise.
            if [[ "$ip_local" != *"$host"* && ("$host" == 192.168.* || "$host" == 10.*) ]]; then
                # Accessing remote host via ssh and retrieving the number of connected
                # compute-capable devices in the host.
                gpu_local=$(sshrun $host 2> /dev/null)

                # If the number of devices found is greater than zero, than we can use
                # this host to execute our code.
                if [[ "$gpu_local" -gt 0 ]]; then
                    echo "$host slots=$gpu_local max-slots=$gpu_local" >> "$1"
                    gpu_total=$((gpu_local + gpu_total))
                fi
            fi

            # Updates the execution progress counter.
            loop_count=$((loop_count + 1))
        done

        # Notifies the watchdog about the conclusion of this task's execution.
        wnotify finish "hostfinder" "found <bold>$gpu_total</> compute-capable devices"
    }

    # Defines the default file name if none given.
    hostfile=".hostfile"

    # Discovers the name of hostfile to be used. If no valid file name is given,
    # the default name is used.
    if [[ ! -z "$1" && ! (-e "$1" && ! -f "$1") ]]; then
        hostfile="$1"
    fi

    # Generates the hostfile.
    generate "$hostfile"
fi
