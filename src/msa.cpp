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
 * Declaring aliases for output file descriptions.
 */
static std::ostream& out = std::cout;
static std::ostream& err = std::cerr;

/**
 * The application class. This class is responsible for running the application's
 * specific functions.
 * @since 0.1.alpha
 */
class Application final
{

    private:
        Fasta fasta;            /// The Fasta file to be aligned.
        Pairwise pwise;         /// The pairwise step instance.

    protected:
        Timer<> timer;          /// The timer to use when measuring execution times.

    public:
        Application() = default;

        /**
         * Runs the application. This method times the execution of all steps for
         * the multiple sequence alignment.
         */
        void run()
        {
            report("total", this->timer.run([this]() {
                report("loading", this->timer.run([this]() { this->load(); }));
                report("pairwise", this->timer.run([this]() { this->pairwise(); }));
            }));
        }

    private:
        /**
         * Loads the Fasta file, the first step. The sequences will be loaded into
         * the master node and broadcasted to all other nodes.
         */
        void load()
        {
            this->fasta = Fasta(cmd.get("filename"));
            Fasta::broadcast(this->fasta);
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
                out << s_bold "[ report]: " s_reset
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

    cmd.init({
        {"h", "help",     "Displays this help menu." }
    ,   {"v", "version",  "Displays the software version." }
    ,   {"m", "multigpu", "Try to use multiple devices in a single host."}
    ,   {"f", "file",     "File to be processed.", "filename"}
    ,   {"x", "matrix",   "Choose the scoring matrix to use.", "matrix"}
    }, {"filename"});

    cmd.parse(argc, argv);
    if(cmd.has("help"))    { onlymaster usage();   exit(0); }
    if(cmd.has("version")) { onlymaster version(); exit(0); }
    cmd.check();

    if(cluster::size < 2)
        finalize({"at least 2 nodes are needed."});

    onlyslaves if(!device::exists())
        finalize(DeviceError::noGPU());

    onlyslaves device::select();
    cluster::sync();

    Application app;
    app.run();

    cluster::finalize();
    return 0;
}
