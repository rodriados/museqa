/** 
 * Multiple Sequence Alignment configuration header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef CONFIG_H_INCLUDED
#define CONFIG_H_INCLUDED

#pragma once

/*
 * Leave uncommented if compiling in debug mode. This may affect many aspects
 * of the software, such as error reporting.
 */
#define msa_debug

/*
 * The software's information. Any of the provided information piece can be printed
 * from the command line as an argument.
 */
#define msa_appname "msa"
#define msa_version "0.1.alpha"
#define msa_author  "Rodrigo Albuquerque de Oliveira Siqueira"
#define msa_email   "rodriados@gmail.com"

/* 
 * Checks whether the system we are compiling in is POSIX compatible. If it
 * is not POSIX compatible, some conditional compiling may take place.
 */
#if defined(unix) || defined(__unix__) || defined(__unix) || defined(__linux__)
#define msa_posix
#define msa_unix
#elif defined(__APPLE__) && defined(__MACH__)
#define msa_posix
#define msa_apple
#elif defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#define msa_windows
#endif

#ifdef msa_windows
#error MSA is not currently compatible with Windows.
#endif

#endif