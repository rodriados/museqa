/**
 * Multiple Sequence Alignment command line processing file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <iostream>
#include <sstream>
#include <vector>
#include <string>

#include "msa.hpp"
#include "input.hpp"

/*
 * Defining the list of commands available from the command line.
 */
const std::vector<Input::Command> Input::commands = {
    Command(ParamCode::Help,     "h", "help",     "Displays this help menu.")
,   Command(ParamCode::Version,  "v", "version",  "Displays the version information.")
,   Command(ParamCode::Verbose,  "b", "verbose",  "Activates the verbose mode.")
,   Command(ParamCode::MultiGPU, "m", "multigpu", "Use multiple GPU devices if possible.")
,   Command(ParamCode::File,     "f", "file",     "File to be loaded into application.", true)
,   Command(ParamCode::Matrix,   "x", "matrix",   "Inform the scoring matrix to use.")
};

/**
 * Parses the command line parameters and organize them for later.
 * @param argc The number of command line arguments.
 * @param argv The command line arguments.
 */
void Input::parse(int argc, const char **argv)
{
    this->appname = argv[0];
    Argument *argument = nullptr;

    for(int i = 1; i < argc; ++i) {
        bool isarg = argv[i][0] == '-';
        const Command& command = isarg ? this->find(argv[i]) : Command::unknown();

        if(!command.is(ParamCode::Unknown))
            argument = this->arguments.find(command.id) == this->arguments.end()
                ? &(this->arguments[command.id] = command)
                : &(this->arguments[command.id]);
        else if(isarg)
            this->unknown(argv[i]);
        else if(argument != nullptr)
            argument->push(argv[i]);
        else
            this->ordered.push_back(argv[i]);
    }

    if(this->arguments.find(ParamCode::Help) != this->arguments.end())
        this->usage();

    if(this->arguments.find(ParamCode::Version) != this->arguments.end())
        this->version();

    for(const Command& command : Input::commands)
        if(command.required && this->arguments.find(command.id) == this->arguments.end())
            this->missing(command);
}

/**
 * Searches for a command by one of its names.
 * @param name The name being searched for.
 * @return The corresponding command for the given name.
 */
const Input::Command& Input::find(const std::string& name) const
{
    for(const Command& command : Input::commands)
        if(command.is(name))
            return command;

    return Command::unknown();
}

/**
 * Builds a unnamed command.
 * @param id The command identifier.
 */
Input::Command::Command(ParamCode id)
:   id(id)
,   required(false)
{}

/**
 * Builds a command from its names and description.
 * @param id The command's identifier.
 * @param sname The command's short name.
 * @param lname The command's long name.
 * @param description The command's description.
 * @param required Is the command required?
 */
Input::Command::Command
(   ParamCode id
,   const std::string& sname
,   const std::string& lname
,   const std::string& description
,   bool required
)
:   id(id)
,   sname(sname.size() > 0 ?  "-" + sname : "")
,   lname(lname.size() > 0 ? "--" + lname : "")
,   description(description)
,   required(required)
{}

/**
 * Builds an argument from a given command.
 * @param command The command to be represented by this argument.
 */
Input::Argument::Argument(const Command& command)
:   command(&command)
,   params({})
{}

/**
 * Prints out a message for missing command arguments.
 * @param command The required command that is missing.
 */
[[noreturn]] void Input::missing(const Command& command) const
{
    __onlymaster {
        std::cerr
            << "Fatal error. The required parameter " __bold
            << command.lname << __reset " was not found." << std::endl
            << "Try `" __bold << this->appname << __reset " -h' for more information." << std::endl;
    }

    finalize(ErrorCode::Success);
}

/**
 * Prints out a message for an unknown command.
 * @param command The unknown command name.
 */
[[noreturn]] void Input::unknown(const char *command) const
{
    __onlymaster {
        std::cerr
            << "Unknown option: " __bold __redfg << command << __reset << std::endl
            << "Try `" __bold << this->appname << __reset " -h' for more information." << std::endl;
    }

    finalize(ErrorCode::Success);
}

/**
 * Prints out the software's current version.
 */
[[noreturn]] void Input::version() const
{
    __onlymaster {
        std::cerr
            << __bold MSA __greenfg " v" VERSION __reset
            << std::endl;
    }

    finalize(ErrorCode::Success);
}

/**
 * Prints out usage guidelines and helps the user to learn the software's commands.
 */
[[noreturn]] void Input::usage() const
{
    __onlymaster {
        std::cerr
            << "Usage: mpirun " __bold << this->appname << __reset " [options]" << std::endl
            << "Options:" << std::endl;

        for(const Command& command : Input::commands)
            std::cerr
                << "  " __bold << command.sname << ", " << command.lname << __reset
                << (command.required ? " (required)" : "") << std::endl
                << "    " << command.description << std::endl << std::endl;
    }

    finalize(ErrorCode::Success);
}
