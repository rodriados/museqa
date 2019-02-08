# Multiple Sequence Alignment makefile.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018 Rodrigo Siqueira
NAME = msa

INCDIR = inc
SRCDIR = src
OBJDIR = obj
TESTDIR = test

MPCC ?= mpicc
MPPP ?= mpic++
NVCC ?= nvcc
PYCC ?= g++
PYXC ?= cython

# Target architecture for CUDA compilation. This indicates the minimum
# support required for the codebase.
NVARCH ?= sm_30

# Defining language standards to be used. These can be overriden by
# environment variables.
STDC   ?= c99
STDCPP ?= c++14
STDCU  ?= c++11

MPILIBDIR ?= /usr/lib/openmpi/lib
PY2INCDIR ?= /usr/include/python2.7

# Defining macros inside code at compile time. This can be used to enable
# or disable certain marked features on code.
DEFS ?= 

MPCCFLAGS = -std=$(STDC) -I$(INCDIR) -g -Wall -lm -fPIC $(DEFS)
MPPPFLAGS = -std=$(STDCPP) -I$(INCDIR) -g -Wall -fPIC $(DEFS)
NVCCFLAGS = -std=$(STDCU) -I$(INCDIR) -g -arch $(NVARCH) -lmpi -lcuda -lcudart -w $(DEFS)
PYCCFLAGS = -std=$(STDCPP) -I$(INCDIR) -I$(PY2INCDIR) -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing $(DEFS)
PYXCFLAGS = --cplus -I$(INCDIR)
LINKFLAGS = -L$(MPILIBDIR) -lmpi_cxx -lmpi -g

# Lists all files to be compiled and separates them according to their
# corresponding compilers.
MPCCFILES := $(shell find $(SRCDIR) -name '*.c')
MPPPFILES := $(shell find $(SRCDIR) -name '*.cpp')
NVCCFILES := $(shell find $(SRCDIR) -name '*.cu')
PYXCFILES := $(shell find $(SRCDIR) -name '*.pyx')
TDEPFILES := $(shell find $(TESTDIR)/$(NAME) -name '*.d')

SRCINTERNAL = $(sort $(dir $(wildcard $(SRCDIR)/*/. $(SRCDIR)/*/*/.)))
OBJINTERNAL = $(SRCINTERNAL:$(SRCDIR)/%=$(OBJDIR)/%)
TSTINTERNAL = $(SRCINTERNAL:$(SRCDIR)/%=$(TESTDIR)/$(NAME)/%)

ODEPS = $(MPCCFILES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)                            \
        $(MPPPFILES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)                          \
        $(NVCCFILES:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)
TDEPS = $(PYXCFILES:$(SRCDIR)/%.pyx=$(OBJDIR)/%.pyx.o)                      \
        $(PYXCFILES:$(SRCDIR)/%.pyx=$(TESTDIR)/$(NAME)/%.so)
HDEPS = $(ODEPS:$(OBJDIR)/%.o=$(OBJDIR)/%.d)

all: production

install:
	@mkdir -p $(OBJINTERNAL)
	@mkdir -p $(TSTINTERNAL)

production: install $(OBJDIR)/$(NAME)
	@chmod +x src/watchdog.sh
	@chmod +x hostfinder
	@chmod +x msarun

testing: override MPPP = $(PYCC)
testing: override DEFS = -Dmsa_compile_cython
testing: install $(TDEPS)

clean:
	@rm -rf $(NAME) $(OBJDIR) $(SRCDIR)/*~ *~
	@rm -rf $(TSTINTERNAL) $(TESTDIR)/$(NAME)/*.so $(TESTDIR)/$(NAME)/*.pyc

$(OBJDIR)/$(NAME): $(ODEPS)
	$(NVCC) $(LINKFLAGS) $^ -o $@

# Creates dependency on header files. This is valuable so that whenever
# a header file is changed, all objects depending on it will be recompiled.
-include $(HDEPS) $(TDEPFILES)

# Compiling C files.
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(MPCC) $(MPCCFLAGS) -MMD -c $< -o $@

# Compiling C++ files.
$(OBJDIR)/%.o $(OBJDIR)/%.pyx.o: $(SRCDIR)/%.cpp
	$(MPPP) $(MPPPFLAGS) -MMD -c $< -o $@

# Compiling CUDA files.
$(OBJDIR)/%.o $(OBJDIR)/%.pyx.o: $(SRCDIR)/%.cu
	@$(NVCC) $(NVCCFLAGS) -M $< -odir $(patsubst %/,%,$(dir $@)) > $(@:%.o=%.d)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# If no correspondent file has been found, simply ignore.
$(OBJDIR)/%.pyx.o: ;

# Converting Cython files to C++
$(OBJDIR)/%.cxx: $(SRCDIR)/%.pyx
	$(PYXC) $(PYXCFLAGS) $< -o $@

# Compiling Cython C++ files to Python modules
.SECONDEXPANSION:
$(TESTDIR)/$(NAME)/%.so: $(OBJDIR)/%.cxx $$(wildcard $(OBJDIR)/%.pyx.o)
	$(PYCC) $(PYCCFLAGS) $^ -o $@

.PHONY: all clean install production testing

.PRECIOUS: $(OBJDIR)/%.cxx $(OBJDIR)/%.pyx.o
