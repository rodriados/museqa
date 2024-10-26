# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file The software's compilation and instalation script.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-present Rodrigo Siqueira
NAME = museqa

INCDIR  = src
SRCDIR  = src
TESTDIR = test

DSTDIR ?= dist
OBJDIR ?= obj
BINDIR ?= bin
PT3DIR ?= thirdparty

GCCC   ?= mpicc
GCPP   ?= mpic++
NVCC   ?= nvcc
STDC   ?= c99
STDCPP ?= c++17
STDCU  ?= c++17

# Target architecture for CUDA compilation. This indicates the minimum support required
# for the codebase but can be changed with environment variables.
NVARCH ?= sm_30

# Defining macros inside code at compile time. This can be used to enable or disable
# certain features on code or affect the projects compilation.
FLAGS ?=
GCPPFLAGS ?= -std=$(STDCPP) -I$(DSTDIR) -I$(INCDIR) $(FLAGS)
LINKFLAGS ?= $(FLAGS)

# The operational system check. At least for now, we assume that we are always running
# on a Linux machine. Therefore, a disclaimer must be shown if this is not true.
SYSTEMOS := $(shell uname)
SYSTEMOS := $(patsubst MINGW%,Windows,$(SYSTEMOS))
SYSTEMOS := $(patsubst MSYS%,Msys,$(SYSTEMOS))
SYSTEMOS := $(patsubst CYGWIN%,Msys,$(SYSTEMOS))

ifneq ($(SYSTEMOS), Linux)
  $(info Warning: This makefile assumes OS to be Linux.)
endif

# If running an installation target, a prefix variable is used to determine where
# the files must be copied to. In this context, a default value must be provided.
ifeq ($(PREFIX),)
	PREFIX := /usr/local
endif

all: distribute

prepare-distribute:
	@mkdir -p $(DSTDIR)

export DISTRIBUTE_DESTINATION ?= $(shell realpath $(DSTDIR))

distribute: prepare-distribute thirdparty-distribute
no-thirdparty-distribute: prepare-distribute

clean-distribute: thirdparty-clean
	@rm -rf $(DSTDIR)

clean: clean-distribute
	@rm -rf $(BINDIR)
	@rm -rf $(OBJDIR)

.PHONY: all clean
.PHONY: prepare-distribute distribute no-thirdparty-distribute clean-distribute

# The target path for third party dependencies' distribution files. As each dependency
# may allow different settings, a variable for each one is needed.
THIRDPARTY_IGNORE ?=
THIRDPARTY_DEPENDENCIES = mpiwcpp17 reflector supertuple

THIRDPARTY_TARGETS := $(filter-out $(THIRDPARTY_IGNORE),$(THIRDPARTY_DEPENDENCIES))
THIRDPARTY_TARGETS := $(THIRDPARTY_TARGETS:%=$(DISTRIBUTE_DESTINATION)/%.h)

thirdparty-distribute: prepare-distribute $(THIRDPARTY_TARGETS)
thirdparty-install:    $(THIRDPARTY_DEPENDENCIES:%=thirdparty-install-%)
thirdparty-uninstall:  $(THIRDPARTY_DEPENDENCIES:%=thirdparty-uninstall-%)
thirdparty-clean:      $(THIRDPARTY_DEPENDENCIES:%=thirdparty-clean-%)

ifndef MUSEQA_DIST_STANDALONE

export SUPERTUPLE_DIST_STANDALONE = 1
export REFLECTOR_DIST_STANDALONE  = 1
export MPIWCPP17_DIST_STANDALONE  = 1

thirdparty-distribute-%: $(DISTRIBUTE_DESTINATION)/%.h

$(DISTRIBUTE_DESTINATION)/%.h: %
	@$(MAKE) --no-print-directory -C $(PT3DIR)/$< distribute

thirdparty-install-%: %
	@$(MAKE) --no-print-directory -C $(PT3DIR)/$< install

thirdparty-uninstall-%: %
	@$(MAKE) --no-print-directory -C $(PT3DIR)/$< uninstall

thirdparty-clean-%: %
	@$(MAKE) --no-print-directory -C $(PT3DIR)/$< clean

else
.PHONY: $(THIRDPARTY_TARGETS)
.PHONY: $(THIRDPARTY_DEPENDENCIES:%=thirdparty-distribute-%)
.PHONY: $(THIRDPARTY_DEPENDENCIES:%=thirdparty-install-%)
.PHONY: $(THIRDPARTY_DEPENDENCIES:%=thirdparty-uninstall-%)
.PHONY: $(THIRDPARTY_DEPENDENCIES:%=thirdparty-clean-%)
endif

.PHONY: thirdparty-distribute thirdparty-install thirdparty-uninstall thirdparty-clean
.PHONY: $(THIRDPARTY_DEPENDENCIES)
