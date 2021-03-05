# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file The software's compilation and instalation script.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-present Rodrigo Siqueira

# Let's allow the user to easily include some definitions and variable overrides
# if they have the need. Execution should not fail if this file doesn't exist.
-include user.make

ifndef build
	build := release
endif

default: all

# Including the complete building rules for the project's code. These rules will
# compile all needed code to produce an executable program at the end.
include make/rules.make
include make/thirdparty.make


# @@@ REFERENCE
# http://git.ghostscript.com/?p=mupdf.git;a=blob;f=Makethird;h=db1eeb5ae8a632155b8e61bd9a6cba5ba0f3d428;hb=HEAD

WATCHLIST = $(shell find src -type f \( -name "*.c*" -o -name "*.h*" \))

watch:
	@inotifywait -q -e modify $(WATCHLIST)

watch-recompile:
	@while ! inotifywait -q -e modify $(WATCHLIST); do time -p $(MAKE); done

.PHONY: all watch watch-recompile
