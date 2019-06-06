/**
 * Multiple Sequence Alignment command line processing file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <string>
#include <vector>
#include <map>

#include "cmdline.hpp"
#include "exception.hpp"

/**
 * The global command line parser instance.
 * @since 0.1.1
 */
cmdline::Parser cmdline::parser;

/**
 * Initializes the parser with the options it should parse.
 * @param options The list of available options for this parser.
 */
void cmdline::Parser::init(const std::vector<cmdline::Option>& options)
{
    for(const cmdline::Option& option : options) {
        if(option.isRequired())
            required.push_back(option.getLname());

        this->options["-"  + option.getSname()] = option;
        this->options["--" + option.getLname()] = option;
    }
}

/**
 * Parses the command line arguments through its options.
 * @param argc The number of command line arguments.
 * @param argv The command line arguments.
 */
void cmdline::Parser::parse(int argc, char **argv)
{
    for(int i = 1; i < argc; ++i) {
        const cmdline::Option& option = find(argv[i]);

        if(!option.isUnknown() && option.isVariadic()) {
            enforce(i + 1 >= argc, "unknown option: {}", option.getLname());

            values[option.getLname()] = argv[++i];
            continue;
        }

        if(!option.isUnknown()) {
            values[option.getLname()] = option.getLname();
            continue;
        }

        positional.push_back(argv[i]);
    }

    for(const std::string& option : required)
        enforce(has(option), "missing option: {}", option);

    appname = argv[0];
}

/**
 * Searches for an option via one of its names.
 * @param needle The option being searched for.
 * @return The found option or an unknown option.
 */
const cmdline::Option& cmdline::Parser::find(const std::string& needle) const
{
    static cmdline::Option unknown {};
    const auto& value = options.find(needle);

    return value != options.end()
        ? value->second
        : unknown;
}