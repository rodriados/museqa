/** 
 * Multiple Sequence Alignment input header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _INPUT_HPP_
#define _INPUT_HPP_

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

/** 
 * Lists all possible parameter codes from the terminal. These codes are used
 * as an interface when requiring the arguments's values.
 * @since 0.1.alpha
 */
enum class ParamCode : uint8_t
{
    Unknown = 0
,   Help
,   Version
,   Verbose
,   File
,   MultiGPU
,   Matrix
};

/**
 * Parses the parameters given from the command line, and stores them so it can
 * be easily retrieved when needed.
 * @since 0.1.alpha
 */
class Input
{
    private:
        /**
         * Stores all information about a given command available from the command
         * line. There should be an instance for each command.
         * @since 0.1.alpha
         */
        class Command
        {
            public:
                ParamCode id;               /// The command's identifier.
                std::string sname;          /// The command's short name option.
                std::string lname;          /// The command's long name option.
                std::string description;    /// The command's description.
                const bool variadic;        /// Does the command receive any parameter?
                const bool required;        /// Is the command required?

            public:
                explicit Command
                    (   ParamCode
                    ,   const std::string&
                    ,   const std::string&
                    ,   const std::string&
                    ,   bool = false
                    ,   bool = false
                    );
                virtual ~Command() = default;

                /**
                 * Checks whether the command is the one being requested.
                 * @param id The identifier of the requested command.
                 * @return Is this command the requested one?
                 */
                inline bool is(ParamCode id) const
                {
                    return id == this->id;
                }

                /**
                 * Checks whether the command is the one being requested from names.
                 * @param given The name of the requested command.
                 * @return Is this command the requested one?
                 */
                inline bool is(const std::string& given) const
                {
                    return given == this->lname || given == this->sname;
                }

                /**
                 * Provides a static command instance for an unknown command.
                 * @return The unknown command instance.
                 */
                inline static const Command& unknown()
                {
                    static Command unknown(ParamCode::Unknown);
                    return unknown;
                }

            private:
                explicit Command(ParamCode);
        };

        /**
         * Represents an argument given by the command line. An argument can hold
         * multiple parameters. That means that positional arguments must be given
         * before any named parameter.
         * @since 0.1.alpha
         */        
        class Argument
        {
            public:
                const Command *command;             /// Pointer to the command being represented.
                std::string value;                  /// The parameter given to the argument.

            public:
                Argument() = default;
                explicit Argument(const Command&);

                /**
                 * Checks whether the argument is the one being requested.
                 * @param id The identifier of the requested argument.
                 * @return Is this argument the requested one?
                 */
                inline bool is(ParamCode id) const
                {
                    return this->command->id == id;
                }

                /**
                 * Retrieves the value associated to the command.
                 * @return The argument's value.
                 */
                inline const std::string& get() const
                {
                    return this->value;
                }

                /**
                 * Provides a static argument instance for an unknown argument.
                 * @return The unknown argument instance.
                 */
                inline static const Argument& unknown()
                {
                    static Argument unknown(Command::unknown());
                    return unknown;
                }

            private:
                /**
                 * Allows the command represented by this argument to change.
                 * This is only needed by the parent class when parsing.
                 * @param command The new command being represented.
                 * @return This argument instance.
                 */
                inline Argument& operator=(const Command& command)
                {
                    this->command = &command;
                    return *this;
                }

                /**
                 * Sets a new value to the argument.
                 * @param value The parameter sent as parameter to this argument.
                 */
                inline void set(const char *value)
                {
                    this->value = std::string(value);
                }

            friend class Input;
        };

    private:
        std::string appname;                        /// The name used by the application.
        std::vector<std::string> ordered;           /// The unnamed arguments.
        std::map<ParamCode, Argument> arguments;    /// The map of named arguments.

        static const std::vector<Command> commands; /// The list of commands available.

    public:
        Input() = default;

        /**
         * Checks whether an ordered argument exists.
         * @param offset The requested argument.
         * @return Does the argument exist?
         */
        inline bool has(unsigned int offset) const
        {
            return this->ordered.size() > offset;
        }

        /**
         * Checks whether an named argument exists.
         * @param id The identifier of requested argument.
         * @return Does the argument exist?
         */
        inline bool has(ParamCode id) const
        {
            for(const auto& argument : this->arguments)
                if(argument.second.is(id))
                    return true;

            return false;
        }

        /**
         * Retrieves an ordered argument.
         * @param offset The offset being requested.
         * @return The value of requested parameter.
         */
        inline const std::string& get(int offset) const
        {
            return this->ordered.at(offset);
        }

        /**
         * Retrieves a named argument.
         * @param id The identifier of requested argument.
         * @param def The value to be returned if not found.
         * @return The requested argument instance.
         */
        inline const std::string& get(ParamCode id) const
        {
            for(const auto& argument : this->arguments)
                if(argument.second.is(id))
                    return argument.second.get();

            static const std::string null;
            
            return null;
        }

        void parse(int, char **);
        void checkhelp() const;

    private:
        const Command& find(const std::string&) const;

        [[noreturn]] void missing(const Command&) const;
        [[noreturn]] void unknown(const char *) const;
        [[noreturn]] void version() const;
        [[noreturn]] void usage() const;
};

extern Input cmd;

#endif