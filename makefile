# Multiple Sequence Alignment makefile.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2020 Rodrigo Siqueira
NAME = msa

INCDIR  = inc
SRCDIR  = src
OBJDIR  = obj
TGTDIR  = bin
TESTDIR = test

GCCC ?= mpicc
GC++ ?= mpic++
NVCC ?= nvcc
PY++ ?= g++
PYXC ?= cython

# Target architecture for CUDA compilation. This indicates the minimum support required
# for the codebase but can be changed with environment variables.
NVARCH ?= sm_30

# Defining language standards to be used. These can be overriden by environment
# variables. We recommend using the default settings, though.
STDC   ?= c99
STDC++ ?= c++14
STDCU  ?= c++11

MPILIBDIR ?= /usr/lib/openmpi/lib
PY3INCDIR ?= /usr/include/python3.5
MPILKFLAG ?= -lmpi_cxx -lmpi

# Defining macros inside code at compile time. This can be used to enable or disable
# certain features on code or affect the projects compilation.
FLAGS ?=

GCCCFLAGS = -std=$(STDC) -I$(INCDIR) -Wall -lm -fPIC -O3 $(FLAGS) $(DEBUG)
GC++FLAGS = -std=$(STDC++) -I$(INCDIR) -Wall -fPIC -O3 $(FLAGS) $(DEBUG)
NVCCFLAGS = -std=$(STDCU) -I$(INCDIR) -arch $(NVARCH) -lmpi -lcuda -lcudart -w -O3 -Xptxas -O3      \
        -Xcompiler -O3 -D_MWAITXINTRIN_H_INCLUDED $(FLAGS) $(DEBUG)
PY++FLAGS = -std=$(STDC++) -I$(INCDIR) -I$(SRCDIR) -I$(PY3INCDIR) -shared -pthread -fPIC -fwrapv    \
        -O2 -Wall -fno-strict-aliasing $(FLAGS) $(DEBUG)
PYXCFLAGS = --cplus -I$(INCDIR) -I$(SRCDIR) -3
LINKFLAGS = -L$(MPILIBDIR) $(MPILKFLAG) $(FLAGS) $(DEBUG)

# Lists all files to be compiled and separates them according to their corresponding
# compilers. Changes in any of these files in will trigger conditional recompilation.
GCCCFILES := $(shell find $(SRCDIR) -name '*.c')
GC++FILES := $(shell find $(SRCDIR) -name '*.cpp')
NVCCFILES := $(shell find $(SRCDIR) -name '*.cu')
PYXCFILES := $(shell find $(SRCDIR) -name '*.pyx')
PYFILES   := $(shell find $(SRCDIR) -name '*.py')

OBJFILES     = $(GCCCFILES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)                                             \
               $(GC++FILES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)                                           \
               $(NVCCFILES:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)
TESTFILES    = $(PYXCFILES:$(SRCDIR)/%.pyx=$(TGTDIR)/%.so)                                          \
               $(PYFILES:$(SRCDIR)/%.py=$(TGTDIR)/%.py)
STATICFILES  = $(filter $(PYXCFILES:$(SRCDIR)/%.pyx=$(OBJDIR)/%.pya.a),$(OBJFILES:%.o=%.pya.a))

SRCHIERARCHY = $(sort $(dir $(wildcard $(SRCDIR)/*/. $(SRCDIR)/*/*/.)))
OBJHIERARCHY = $(SRCHIERARCHY:$(SRCDIR)/%=$(OBJDIR)/%)

all: debug

install:
	@mkdir -p $(OBJHIERARCHY)
	@chmod +x src/hostfinder.sh
	@chmod +x src/watchdog.sh
	@chmod +x msarun

production: install $(TGTDIR)/$(NAME)

debug: override DEBUG = -g
debug: install $(TGTDIR)/$(NAME)

testing: override GC++ = $(PY++)
testing: override DEBUG = -g -Dmsa_target_cython
testing: override NVCCFLAGS = -std=$(STDCU) -I$(INCDIR) -g -arch $(NVARCH) -lcuda -lcudart -w       \
        -D_MWAITXINTRIN_H_INCLUDED $(DEBUG) --compiler-options -fPIC
testing: install $(TESTFILES) $(TESTDIR)/$(NAME)

clean:
	@rm -rf $(TESTDIR)/$(NAME)
	@rm -rf $(OBJDIR) $(SRCDIR)/*~ *~
	@rm -rf $(TGTDIR)/*.so $(TGTDIR)/$(NAME)

# Creates dependency on header files. This is valuable so that whenever a header
# file is changed, all objects depending on it will be recompiled.
ifneq ($(wildcard $(OBJDIR)/.),)
-include testing.d $(shell find $(OBJDIR) -name '*.d')
endif

# The step rules to generate the main production executable. These rules can be
# parallelized to improve compiling time.
$(TGTDIR)/$(NAME): $(OBJFILES)
	$(NVCC) $(LINKFLAGS) $^ -o $@

$(OBJDIR)/%.o $(OBJDIR)/%.pya.a: $(SRCDIR)/%.c
	$(GCCC) $(GCCCFLAGS) -MMD -c $< -o $@

$(OBJDIR)/%.o $(OBJDIR)/%.pya.a: $(SRCDIR)/%.cpp
	$(GC++) $(GC++FLAGS) -MMD -c $< -o $@

$(OBJDIR)/%.o $(OBJDIR)/%.pya.a: $(SRCDIR)/%.cu
	@$(NVCC) $(NVCCFLAGS) -M $< -odir $(patsubst %/,%,$(dir $@)) > $(@:%.o=%.d)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

# The step rules to build the testing environment modules. This setting will result
# into many modules that can be directly imported used into a python module.
$(TGTDIR)/%.so: $(OBJDIR)/%.pyo.so $(OBJDIR)/libmodules.a
	$(NVCC) -shared -L$(OBJDIR) -lmodules $< -o $@

$(TGTDIR)/%.py: $(SRCDIR)/%.py
	cp $< $@

$(OBJDIR)/%.cxx: $(SRCDIR)/%.pyx $(INCDIR)/*.pxd
	$(PYXC) $(PYXCFLAGS) $< -o $@

$(OBJDIR)/%.pyo.so: $(OBJDIR)/%.cxx
	$(PY++) $(PY++FLAGS) -MMD -c $< -o $@

$(OBJDIR)/libmodules.a:                                                                             \
    $(OBJDIR)/encoder.pya.a                                                                         \
    $(OBJDIR)/parser/fasta.pya.a

$(OBJDIR)/libmodules.a: $(STATICFILES)
	ar rcs $@ $^

$(TESTDIR)/$(NAME):
	@ln -r -s $(TGTDIR) $@

.PHONY: all install production debug testing clean

.PRECIOUS: $(OBJDIR)/%.cxx $(OBJDIR)/%.pyo.so $(OBJDIR)/%.pya.a
