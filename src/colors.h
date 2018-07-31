/** 
 * Multiple Sequence Alignment color macros header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _COLORS_H_
#define _COLORS_H_

/*
 * Defining foreground colors.
 */
#define __blackfg  	"\033[30m"
#define __redfg    	"\033[31m"
#define __greenfg  	"\033[32m"
#define __yellowfg 	"\033[33m"
#define __bluefg   	"\033[34m"
#define __magentafg	"\033[35m"
#define __cyanfg   	"\033[36m"
#define __whitefg  	"\033[37m"
#define __normalfg 	"\033[39m"

/*
 * Defining background colors.
 */
#define __blackbg  	"\033[40m"
#define __redbg    	"\033[41m"
#define __greenbg  	"\033[42m"
#define __yellowbg 	"\033[43m"
#define __bluebg   	"\033[44m"
#define __magentabg	"\033[45m"
#define __cyanbg   	"\033[46m"
#define __whitebg  	"\033[47m"
#define __normalbg 	"\033[49m"

/*
 * Defining text styles.
 */
#define __bold     	"\033[1m"
#define __dim      	"\033[2m"
#define __normal   	"\033[22m"
#define __reset    	"\033[0m"
#define __underline	"\033[4m"
#define __blinkslow	"\033[5m"
#define __blinkfast	"\033[6m"
#define __italic   	"\033[3m"
#define __inverted 	"\033[7m"

#define fg(color, msg)     __##color##fg msg __reset
#define bg(color, msg)     __##color##bg msg __reset
#define style(style, msg)  __##style     msg __reset

#endif