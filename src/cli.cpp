/**
 * Multiple Sequence Alignment command line processing file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <iostream>
#include <vector>
#include <string>

#include "msa.hpp"
#include "cli.hpp"

CliParser cli;

/**
 * Initializes the parser with the options it should parse.
 * @param options The list of available options for this parser.
 * @param arguments The list of positional (and required) arguments.
 */
void CliParser::init
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
void CliParser::parse(int argc, char **argv)
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
            if(i + 1 >= argc) error("unknown option '%s'", option.getLname().c_str());
            this->values[option.getArgument()] = argv[++i];
            continue;
        }

        if(!option.isUnknown()) {
            this->values[option.getLname()] = option.getLname();
            continue;
        }

        error("unknown option '%s'", argv[i]);
    }

    for(const std::string& required : this->arguments)
        if(!this->has(required))
            error("required option '%s' is missing", required.c_str());
}

/**
 * Searches for an option via one of its names.
 * @param needle The option being searched for.
 * @return The found option or an unknown option.
 */
const Option& CliParser::find(const std::string& needle) const
{
    static Option unknown {};

    for(const Option& option : this->options)
        if("--" + option.getLname() == needle || "-" + option.getSname() == needle)
            return option;

    return unknown;
}
