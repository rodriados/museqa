# Multiple Sequence Alignment makefile.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-2020 Rodrigo Siqueira
NAME = msa

INCDIR  = src
SRCDIR  = src
OBJDIR  = obj
TGTDIR  = bin
TESTDIR = test

GCCC ?= mpicc
GCPP ?= mpic++
NVCC ?= nvcc
PYPP ?= g++
PYXC ?= cython

# Target architecture for CUDA compilation. This indicates the minimum support required
# for the codebase but can be changed with environment variables.
NVARCH ?= sm_30

# Defining language standards to be used. These can be overriden by environment
# variables. We recommend using the default settings, though.
STDC   ?= c99
STDCPP ?= c++14
STDCU  ?= c++11

MPILIBDIR ?= /usr/lib/openmpi/lib
PY3INCDIR ?= /usr/include/python3.5
MPILKFLAG ?= -lmpi_cxx -lmpi

# Defining macros inside code at compile time. This can be used to enable or disable
# certain features on code or affect the projects compilation.
FLAGS ?=

GCCCFLAGS = -std=$(STDC) -I$(INCDIR) -Wall -lm -fPIC -O3 $(FLAGS) $(ENV)
GCPPFLAGS = -std=$(STDCPP) -I$(INCDIR) -Wall -fPIC -O3 $(FLAGS) $(ENV)
NVCCFLAGS = -std=$(STDCU) -I$(INCDIR) -arch $(NVARCH) -lmpi -lcuda -lcudart -w -O3 -Xptxas -O3      \
        -Xcompiler -O3 -D_MWAITXINTRIN_H_INCLUDED $(FLAGS) $(ENV)
PYPPFLAGS = -std=$(STDCPP) -I$(INCDIR) -I$(SRCDIR) -I$(PY3INCDIR) -shared -pthread -fPIC -fwrapv    \
        -O2 -Wall -fno-strict-aliasing $(FLAGS) $(ENV)
PYXCFLAGS = --cplus -I$(INCDIR) -I$(SRCDIR) -3
LINKFLAGS = -L$(MPILIBDIR) $(MPILKFLAG) $(FLAGS) $(ENV)

# Lists all files to be compiled and separates them according to their corresponding
# compilers. Changes in any of these files in will trigger conditional recompilation.
GCCCFILES := $(shell find $(SRCDIR) -name '*.c')
GCPPFILES := $(shell find $(SRCDIR) -name '*.cpp')
NVCCFILES := $(shell find $(SRCDIR) -name '*.cu')
PYXCFILES := $(shell find $(SRCDIR) -name '*.pyx')
PYFILES   := $(shell find $(SRCDIR) -name '*.py')

OBJFILES     = $(GCCCFILES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)                                             \
               $(GCPPFILES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)                                           \
               $(NVCCFILES:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)
TESTFILES    = $(PYXCFILES:$(SRCDIR)/%.pyx=$(TGTDIR)/%.so)                                          \
               $(PYFILES:$(SRCDIR)/%.py=$(TGTDIR)/%.py)
STATICFILES  = $(filter $(PYXCFILES:$(SRCDIR)/%.pyx=$(OBJDIR)/%.pya.a),$(OBJFILES:%.o=%.pya.a))

OBJHIERARCHY = $(sort $(dir $(OBJFILES)))

all: debug

install: $(OBJHIERARCHY)
	@chmod +x src/hostfinder.sh
	@chmod +x src/watchdog.sh
	@chmod +x museqa

production: install
production: override ENV = -DPRODUCTION
production: $(TGTDIR)/$(NAME)

debug: install
debug: override ENV = -g -DDEBUG
debug: $(TGTDIR)/$(NAME)

testing: install
testing: override GCPP = $(PYPP)
testing: override ENV = -g -DTESTING
testing: override NVCCFLAGS = -std=$(STDCU) -I$(INCDIR) -g -arch $(NVARCH) -lcuda -lcudart -w       \
        -D_MWAITXINTRIN_H_INCLUDED $(ENV) --compiler-options -fPIC
testing: $(TESTFILES)

clean:
	@rm -rf $(OBJDIR)
	@rm -rf $(SRCDIR)/*~ *~
	@rm -rf $(TGTDIR)/*.so $(TGTDIR)/$(NAME)

# Creates dependency on header files. This is valuable so that whenever a header
# file is changed, all objects depending on it will be recompiled.
ifneq ($(wildcard $(OBJDIR)/.),)
-include testing.d $(shell find $(OBJDIR) -name '*.d')
endif

# Creates the hierarchy of folders needed to compile the project. This rule should
# be depended upon by every single build.
$(OBJHIERARCHY):
	@mkdir -p $@

# The step rules to generate the main production executable. These rules can be
# parallelized to improve compiling time.
$(TGTDIR)/$(NAME): $(OBJFILES)
	$(NVCC) $(LINKFLAGS) $^ -o $@

$(OBJDIR)/%.o $(OBJDIR)/%.pya.a: $(SRCDIR)/%.c
	$(GCCC) $(GCCCFLAGS) -MMD -c $< -o $@

$(OBJDIR)/%.o $(OBJDIR)/%.pya.a: $(SRCDIR)/%.cpp
	$(GCPP) $(GCPPFLAGS) -MMD -c $< -o $@

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
	$(PYPP) $(PYPPFLAGS) -MMD -c $< -o $@

$(OBJDIR)/libmodules.a: $(OBJDIR)/cuda.pya.a
$(OBJDIR)/libmodules.a: $(OBJDIR)/encoder.pya.a
$(OBJDIR)/libmodules.a: $(OBJDIR)/pairwise/pairwise.pya.a
$(OBJDIR)/libmodules.a: $(OBJDIR)/pairwise/table.pya.a
$(OBJDIR)/libmodules.a: $(OBJDIR)/pairwise/database.pya.a
$(OBJDIR)/libmodules.a: $(OBJDIR)/io/loader/database.pya.a
$(OBJDIR)/libmodules.a: $(OBJDIR)/io/loader/parser/fasta.pya.a
$(OBJDIR)/libmodules.a: $(OBJDIR)/pairwise/needleman/communication.pya.a
$(OBJDIR)/libmodules.a: $(OBJDIR)/pairwise/needleman/hybrid.pya.a
$(OBJDIR)/libmodules.a: $(OBJDIR)/pairwise/needleman/sequential.pya.a
$(OBJDIR)/libmodules.a: $(STATICFILES)
	ar rcs $@ $^

.PHONY: all install production debug testing clean

.PRECIOUS: $(OBJDIR)/%.cxx $(OBJDIR)/%.pyo.so $(OBJDIR)/%.pya.a
