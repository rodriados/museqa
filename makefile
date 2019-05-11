# Multiple Sequence Alignment makefile.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2019 Rodrigo Siqueira
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
PY2INCDIR ?= /usr/include/python3.5
MPILKFLAG ?= -lmpi_cxx -lmpi

# Defining macros inside code at compile time. This can be used to enable
# or disable certain marked features on code.
DEFS ?= 

MPCCFLAGS = -std=$(STDC) -I$(INCDIR) -g -Wall -lm -fPIC -O3 $(DEFS)
MPPPFLAGS = -std=$(STDCPP) -I$(INCDIR) -g -Wall -fPIC -O3 $(DEFS)
NVCCFLAGS = -std=$(STDCU) -I$(INCDIR) -g -arch $(NVARCH) -lmpi -lcuda -lcudart -w                   \
		-D_MWAITXINTRIN_H_INCLUDED $(DEFS)
PYCCFLAGS = -std=$(STDCPP) -I$(INCDIR) -I$(PY2INCDIR) -shared -pthread -fPIC -fwrapv -O2 -Wall      \
		-fno-strict-aliasing $(DEFS)
PYXCFLAGS = --cplus -I$(INCDIR)
LINKFLAGS = -L$(MPILIBDIR) $(MPILKFLAG) -g

# Lists all files to be compiled and separates them according to their
# corresponding compilers.
MPCCFILES := $(shell find $(SRCDIR) -name '*.c')
MPPPFILES := $(shell find $(SRCDIR) -name '*.cpp')
NVCCFILES := $(shell find $(SRCDIR) -name '*.cu')
PYXCFILES := $(shell find $(SRCDIR) -name '*.pyx')
TDEPFILES := $(shell find $(TESTDIR)/$(NAME) -name '*.d')                                           \
             $(shell find $(OBJDIR)/$(TESTDIR) -name '*.d' 2>/dev/null)

SRCINTERNAL = $(sort $(dir $(wildcard $(SRCDIR)/*/. $(SRCDIR)/*/*/.)))
OBJINTERNAL = $(SRCINTERNAL:$(SRCDIR)/%=$(OBJDIR)/%)                                                \
              $(SRCINTERNAL:$(SRCDIR)/%=$(OBJDIR)/$(TESTDIR)/%)

ODEPS = $(MPCCFILES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)                                                    \
        $(MPPPFILES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)                                                  \
        $(NVCCFILES:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)
TDEPS = $(PYXCFILES:$(SRCDIR)/%.pyx=$(OBJDIR)/$(TESTDIR)/%.so)                                      \
        $(PYXCFILES:$(SRCDIR)/%.pyx=$(TESTDIR)/$(NAME)/%.so)
HDEPS = $(ODEPS:$(OBJDIR)/%.o=$(OBJDIR)/%.d)

all: production

install:
	@mkdir -p $(OBJINTERNAL)

production: install $(OBJDIR)/$(NAME)
	@chmod +x src/watchdog.sh
	@chmod +x hostfinder
	@chmod +x msarun

testing: override MPPP = $(PYCC)
testing: override DEFS = -Dmsa_compile_cython
testing: override NVCCFLAGS = -std=$(STDCU) -I$(INCDIR) -g -arch $(NVARCH) -lcuda -lcudart -w       \
							-D_MWAITXINTRIN_H_INCLUDED $(DEFS) --compiler-options -fPIC
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
$(OBJDIR)/%.o $(OBJDIR)/$(TESTDIR)/%.so: $(SRCDIR)/%.cpp
	$(MPPP) $(MPPPFLAGS) -MMD -c $< -o $@

# Compiling CUDA files.
$(OBJDIR)/%.o $(OBJDIR)/$(TESTDIR)/%.so: $(SRCDIR)/%.cu
	@$(NVCC) $(NVCCFLAGS) -M $< -odir $(patsubst %/,%,$(dir $@)) > $(@:%.o=%.d)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Converting Cython files to C++
$(OBJDIR)/$(TESTDIR)/%.cxx: $(SRCDIR)/%.pyx
	$(PYXC) $(PYXCFLAGS) $< -o $@

# Compiling Cython generated files.
$(OBJDIR)/$(TESTDIR)/%.py.o: $(OBJDIR)/$(TESTDIR)/%.cxx
	$(PYCC) $(PYCCFLAGS) -MMD -c $< -o $@

# If no correspondent file has been found, simply ignore.
$(OBJDIR)/$(TESTDIR)/%.so: ;

# Linking Python modules.
# Here, we use nvcc so we can access the GPU from Python modules.
.SECONDEXPANSION:
$(TESTDIR)/$(NAME)/%.so: $(OBJDIR)/$(TESTDIR)/%.py.o $$(wildcard $(OBJDIR)/$(TESTDIR)/%.so)
	$(NVCC) -shared $^ -o $@

.PHONY: all clean install production testing

.PRECIOUS: $(OBJDIR)/$(TESTDIR)/%.cxx $(OBJDIR)/$(TESTDIR)/%.py.o $(OBJDIR)/$(TESTDIR)/%.so
