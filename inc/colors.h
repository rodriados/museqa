/** 
 * Multiple Sequence Alignment color macros header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef COLORS_H_INCLUDED
#define COLORS_H_INCLUDED

#pragma once

/*
 * Defining foreground colors.
 */
#define c_black_fg      "\033[30m"
#define c_red_fg        "\033[31m"
#define c_green_fg      "\033[32m"
#define c_yellow_fg     "\033[33m"
#define c_blue_fg       "\033[34m"
#define c_magenta_fg    "\033[35m"
#define c_cyan_fg       "\033[36m"
#define c_white_fg      "\033[37m"
#define c_normal_fg     "\033[39m"

/*
 * Defining background colors.
 */
#define c_black_bg      "\033[40m"
#define c_red_bg        "\033[41m"
#define c_green_bg      "\033[42m"
#define c_yellow_bg     "\033[43m"
#define c_blue_bg       "\033[44m"
#define c_magenta_bg    "\033[45m"
#define c_cyan_bg       "\033[46m"
#define c_white_bg      "\033[47m"
#define c_normal_bg     "\033[49m"

/*
 * Defining text styles.
 */
#define s_bold         "\033[1m"
#define s_dim          "\033[2m"
#define s_normal       "\033[22m"
#define s_reset        "\033[0m"
#define s_underline    "\033[4m"
#define s_blinkslow    "\033[5m"
#define s_blinkfast    "\033[6m"
#define s_italic       "\033[3m"
#define s_inverted     "\033[7m"

#define fg(color, msg)     c_##color##_fg msg s_reset
#define bg(color, msg)     c_##color##_bg msg s_reset
#define style(style, msg)  s_##style      msg s_reset

#endif