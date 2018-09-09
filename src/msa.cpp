/** 
 * Multiple Sequence Alignment main file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <iostream>
#include <string>
#include <vector>

#include "msa.hpp"
#include "input.hpp"
#include "fasta.hpp"
#include "timer.hpp"
#include "device.cuh"
#include "cluster.hpp"

#include "pairwise.hpp"

/*
 * Declaring global variables.
 */
bool verbose = false;
auto& out = std::cout;
auto& err = std::cerr;

/*
 * Declaring global no-return functions.
 */
[[noreturn]] void usage();
[[noreturn]] void version();

/**
 * The application class. This class is responsible for running the application's
 * specific functions.
 * @since 0.1.alpha
 */
class App final
{
    private:
        Timer<> timer;      /// The timer to use when measuring execution times.

    private:
        Fasta fasta;        /// The Fasta file to be aligned.
        Pairwise pwise;     /// The pairwise step instance.

    public:
        App() = default;

        /**
         * Runs the application.
         */
        void run()
        {
            report("loading", this->timer.run([this](){ this->loadfasta(); }));
            report("pairwise", this->timer.run([this](){ this->pairwise(); }));
        }

    private:
        /**
         * Loads the Fasta file, the first step.
         */
        void loadfasta()
        {
            this->fasta = std::move(Fasta(cmd.get("filename")));
            cluster::sync();
        }

        /**
         * Runs the pairwise step.
         */
        void pairwise()
        {
            this->pwise = std::move(Pairwise::run(this->fasta));
            cluster::sync();
        }

        /**
         * Reports the execution success of a step.
         * @param name The name of executed step.
         * @param bm The benchmark instance.
         */
        static void report(const std::string& name, double elapsed)
        {
            onlymaster {
                out << MSA << s_bold " [report]: " s_reset
                    << s_bold c_green_fg << name << s_reset " in "
                    << elapsed << " seconds" << std::endl;
            }
        }
};

/**
 * Starts, manages and finishes the software's execution.
 * @param argc Number of arguments sent by command line.
 * @param argv The arguments sent by command line.
 * @return The error code for the operating system.
 */
int main(int argc, char **argv)
{
    cluster::init(argc, argv);

    onlyslaves if(!device::exists())
        finalize(DeviceError::noGPU());

    cmd.init({
        {"h", "help",     "Displays this help menu." }
    ,   {"v", "version",  "Displays the software version." }
    ,   {"b", "verbose",  "Activates verbose mode." }
    ,   {"m", "multigpu", "Try to use multiple devices in a single host."}
    ,   {"f", "file",     "File to be processed.", "filename"}
    ,   {"x", "matrix",   "Choose the scoring matrix to use.", "matrix"}
    }, {"filename"});

    cmd.parse(argc, argv);
    if(cmd.has("help"))     usage();
    if(cmd.has("version"))  version();
    cmd.check();

    verbose = cmd.has("verbose");
    cluster::sync();

    App msa;
    msa.run();

    cluster::finalize();
    return 0;
}

/**
 * Prints out a message of help for the user. The message uses the description
 * of options given and set on main.
 * @see main
 */
[[noreturn]] void usage()
{
    onlymaster {
        err << MSA << s_bold " [usage]: " s_reset
            << "mpirun [...] " << cmd.getAppname() << " [options]" << std::endl
            << MSA << s_bold " [options]:" << std::endl;

        for(const Option& option : cmd.getOptions())
            err << s_bold "  -" << option.getSname() << ", --" << option.getLname() << s_reset " "
                << (!option.getArgument().empty() ? option.getArgument() : "") << std::endl
                << "    " << option.getDescription() << std::endl;
    }

    finalize(Error::success());
}

/**
 * Prints out the software's current version. This is important so the user can
 * know whether they really are using the software they want to.
 * @see main
 */
[[noreturn]] void version()
{
    onlymaster {
        err << MSA << s_bold " [version]: " s_reset
            << MSA_VERSION << std::endl;
    }

    finalize(Error::success());
}

/**
 * Aborts the execution and kills all processes.
 * @param error Error detected during execution.
 */
[[noreturn]] void finalize(Error error)
{
    onlymaster if(!error.msg.empty()) {
        err << MSA << s_bold " [fatal error]: " s_reset
            << error.msg << std::endl
            << "execution has terminated." << std::endl;
    }

    cluster::finalize();
    exit(0);
}