/**
 * Multiple Sequence Alignment command line processing file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <iostream>
#include <vector>
#include <string>

#include "msa.hpp"
#include "input.hpp"

/*
 * Declaring global variables and functions.
 */
Parser cmd;
bool verbose = false;

/**
 * Initializes the parser with the options it should parse.
 * @param options The list of available options for this parser.
 * @param arguments The list of positional (and required) arguments.
 */
void Parser::init
    (   const std::vector<Option>& options
    ,   const std::vector<std::string>& arguments   )
{
    this->options = options;
    this->arguments = arguments;
}

/**
 * Parses the command line arguments through its options.
 * @param argc The number of command line arguments.
 * @param argv The command line arguments.
 */
void Parser::parse(int argc, char **argv)
{
    unsigned int position = 0;
    this->appname = argv[0];

    for(int i = 1; i < argc; ++i) {
        const Option& option = (argv[i][0] == '-')
            ? this->find(argv[i])
            : Option {};

        if(option.isUnknown() && this->arguments.size() > position) {
            this->values[this->arguments[position++]] = argv[i];
            continue;
        }

        if(!option.isUnknown() && option.isVariadic()) {
            if(i + 1 >= argc) finalize(InputError::unknown(option.getLname()));
            this->values[option.getArgument()] = argv[++i];
            continue;
        }

        if(!option.isUnknown()) {
            this->values[option.getLname()] = option.getLname();
            continue;
        }

        finalize(InputError::unknown(argv[i]));
    }

    if(this->has("help")) this->usage();
    if(this->has("version")) this->version();
    if(this->has("verbose")) verbose = true;

    for(const std::string& required : this->arguments)
        if(!this->has(required))
            finalize(InputError::missing(required));
}

/**
 * Searches for an option via one of its names.
 * @param needle The option being searched for.
 * @return The found option or an unknown option.
 */
const Option& Parser::find(const std::string& needle) const
{
    static Option unknown {};

    for(const Option& option : this->options)
        if("--" + option.getLname() == needle || "-" + option.getSname() == needle)
            return option;

    return unknown;
}

/**
 * Prints out a message for an unknown command.
 * @param command The unknown command name.
 */
[[noreturn]]
void Parser::usage() const
{
    onlymaster {
        std::cerr
            << "Usage: " s_bold << this->appname << s_reset " [options]" << std::endl
            << "Options:" << std::endl;

        for(const Option& option : this->options)
            std::cerr
                << s_bold "  -" << option.getSname() << ", --" << option.getLname() << s_reset " "
                << (!option.getArgument().empty() ? option.getArgument() : "") << std::endl
                << "    " << option.getDescription() << std::endl << std::endl;
    }

    finalize(Error::success());
}

/**
 * Prints out the software's current version.
 */
[[noreturn]]
void Parser::version() const
{
    onlymaster {
        std::cerr
            << style(bold, "[    msa   ] " c_green_fg "version " MSA_VERSION)
            << std::endl;
    }

    finalize(Error::success());
}

/**
 * Creates an error instance for unknown option.
 * @param option The unknown option.
 * @return The error instance.
 */
const InputError InputError::unknown(const std::string& option)
{
    return InputError("unknown option " s_bold c_red_bg + option + s_reset);
}

/**
 * Creates an error instance for missing option.
 * @param option The missing option.
 * @return The error instance.
 */
const InputError InputError::missing(const std::string& option)
{
    return InputError("required option " s_bold c_green_bg + option + s_reset);
}