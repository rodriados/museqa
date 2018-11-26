/** 
 * Multiple Sequence Alignment main file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <cstdio>
#include <string>
#include <vector>

#include "msa.hpp"
#include "cli.hpp"
#include "fasta.hpp"
#include "timer.hpp"
#include "device.cuh"
#include "cluster.hpp"
#include "pairwise.hpp"

/**
 * The application class. This class is responsible for running the application's
 * specific functions and managing its resources.
 * @since 0.1.alpha
 */
class Application final
{
    protected:
        Fasta fasta;        /// The Fasta file to be aligned.
        Pairwise pwise;     /// The pairwise step instance.

    public:
        /**
         * Runs the application. This method times the execution of all steps for
         * the multiple sequence alignment.
         */
        static void run()
        {
            Timer<> timer;
            Application app;

            report("total", timer.run([&timer, &app]() {
                app.load(cli.get("filename"));
                report("pairwise", timer.run([&app](){ app.pairwise(); }));
                //report("njoining", timer.run([&app](){ njoining(app); }));
                //report("profile-align", timer.run([&app](){ profilealign(app); }));
            }));
        }

    protected:
        /**
         * Loads the Fasta file, the first step. The sequences will be loaded into
         * the master node and broadcasted to all other nodes.
         * @param filename Name of file to load.
         */
        void load(const std::string& filename)
        {
            this->fasta = Fasta(filename);
            cluster::sync();
        }

        /**
         * Runs the pairwise step. This step is responsible for calculating all
         * alignment possibilities between two different sequences.
         */
        void pairwise()
        {
            this->pwise = Pairwise(this->fasta);
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
                printf(s_bold "[report] " c_green_fg "%s" s_reset " done in %.3f seconds\n", name.c_str(), elapsed);
                fflush(stdout);
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

    if(cluster::size < 2)
        error("at least 2 nodes are needed.");

    onlyslaves if(!device::exists())
        error("No compatible GPU has been found.");

    cli.init({
        {"m", "multigpu", "Try to use multiple devices in a single host."}
    ,   {"f", "file",     "File to be processed.", "filename"}
    ,   {"x", "matrix",   "Choose the scoring matrix to use.", "matrix"}
    }, {"filename"});

    cli.parse(argc, argv);

    onlyslaves device::select();
    cluster::sync();

    Application::run();    
    cluster::finalize();

    return 0;
}

/**
 * Quits the software execution with a code.
 * @param code The exit code.
 */
void quit [[noreturn]] (uint8_t code)
{
    cluster::finalize();
    exit(code);
}

/**
 * Prints the software version and quits execution.
 */
void version [[noreturn]] ()
{
    printf("[version] %s\n", msa_version);
    quit();
}
