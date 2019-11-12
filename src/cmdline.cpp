/**
 * Multiple Sequence Alignment command line processing file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <string>
#include <vector>

#include <cmdline.hpp>
#include <exception.hpp>

/**
 * The global command line parser singleton instance.
 * @since 0.1.1
 */
cmdline::parser cmdline::singleton;

/**
 * Initializes the parser with the options it should parse.
 * @param options The list of available options for this parser.
 */
cmdline::parser::parser(const std::vector<cmdline::option>& options) noexcept
{
    for(const cmdline::option& option : options) {
        if(option.required())
            mrequired.push_back(option.longname());

        moptions["-" + option.shortname()] = option;
        moptions["--" + option.longname()] = option;
    }
}

/**
 * Parses the command line arguments through its options.
 * @param argc The number of command line arguments.
 * @param argv The command line arguments.
 */
void cmdline::parser::parse(int argc, char **argv)
{
    for(int i = 1; i < argc; ++i) {
        const cmdline::option& option = find(argv[i]);

        if(!option.empty()) {
            if(option.variadic()) {
                enforce(i + 1 < argc, "missing argument value for '%s'", option.longname());
                mvalues[option.longname()] = argv[++i];
                continue;
            }

            else {
                mvalues[option.longname()] = option.longname();
                continue;
            }
        }

        mpositional.push_back(argv[i]);
    }

    for(const std::string& option : mrequired)
        enforce(has(option), "missing command line argument '%s'", option);

    mappname = argv[0];
}

/**
 * Searches for an option via one of its names.
 * @param needle The option being searched for.
 * @return The found option or an unknown option.
 */
const cmdline::option& cmdline::parser::find(const std::string& needle) const noexcept
{
    static cmdline::option unknown {};

    const auto& value = moptions.find(needle);
    return value != moptions.end() ? value->second : unknown;
}
