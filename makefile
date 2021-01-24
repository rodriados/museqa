# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file The software's compilation and instalation script.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018-present Rodrigo Siqueira
NAME = museqa

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
MPILKFLAG ?= -lmpi_cxx -lmpi
PY3INCDIR ?= $(shell python3 -c "import sysconfig as s; print(s.get_paths()['include'])")

# Defining macros inside code at compile time. This can be used to enable or disable
# certain features on code or affect the projects compilation.
FLAGS ?=

GCCCFLAGS = -std=$(STDC) -I$(INCDIR) -Wall -lm -fPIC -O3 $(ENV) $(FLAGS)
GCPPFLAGS = -std=$(STDCPP) -I$(INCDIR) -Wall -fPIC -O3 $(ENV) $(FLAGS)
NVCCFLAGS = -std=$(STDCU) -I$(INCDIR) -arch $(NVARCH) -lmpi -lcuda -lcudart -w -O3 -Xptxas -O3      \
        -Xcompiler -O3 -D_MWAITXINTRIN_H_INCLUDED $(ENV) $(FLAGS)
PYPPFLAGS = -std=$(STDCPP) -I$(INCDIR) -I$(PY3INCDIR) -shared -pthread -fPIC -fwrapv -O2 -Wall      \
        -fno-strict-aliasing $(ENV) $(FLAGS)
PYXCFLAGS = --cplus -I$(INCDIR) -3
LINKFLAGS = -L$(MPILIBDIR) $(MPILKFLAG) $(ENV) $(FLAGS)

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
TESTFILES    = $(PYXCFILES:$(SRCDIR)/python/%.pyx=$(TGTDIR)/%.so)                                   \
               $(PYFILES:$(SRCDIR)/python/%.py=$(TGTDIR)/%.py)
STATICFILES  = $(filter $(PYXCFILES:$(SRCDIR)/python/%.pyx=$(OBJDIR)/%.a),$(OBJFILES:%.o=%.a))

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
testing: override NVCCFLAGS = -std=$(STDCU) -I$(SRCDIR) -g -arch $(NVARCH) -lcuda -lcudart -w       \
        -D_MWAITXINTRIN_H_INCLUDED $(ENV) --compiler-options -fPIC
testing: $(TESTFILES)

clean:
	@rm -rf $(OBJDIR)
	@rm -rf $(SRCDIR)/*~ *~
	@rm -rf $(TGTDIR)/*.so $(TGTDIR)/*/
	@rm -rf .pytest_cache

# Creates dependency on header files. This is valuable so that whenever a header
# file is changed, all objects depending on it will be recompiled.
ifneq ($(wildcard $(OBJDIR)/.),)
-include $(shell find $(OBJDIR) -name '*.d')
endif

# Creates the hierarchy of folders needed to compile the project. This rule should
# be depended upon by every single build.
$(OBJHIERARCHY):
	@mkdir -p $@

# The step rules to generate the main production executable. These rules can be
# parallelized to improve compiling time.
$(TGTDIR)/$(NAME): $(OBJFILES)
	$(NVCC) $(LINKFLAGS) $^ -o $@

$(OBJDIR)/%.o $(OBJDIR)/%.a: $(SRCDIR)/%.c
	$(GCCC) $(GCCCFLAGS) -MMD -c $< -o $@

$(OBJDIR)/%.o $(OBJDIR)/%.a: $(SRCDIR)/%.cpp
	$(GCPP) $(GCPPFLAGS) -MMD -c $< -o $@

$(OBJDIR)/%.o $(OBJDIR)/%.a: $(SRCDIR)/%.cu
	@$(NVCC) $(NVCCFLAGS) -M $< -odir $(patsubst %/,%,$(dir $@)) > $(@:%.o=%.d)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

# The step rules to build the testing environment modules. This setting will result
# into many modules that can be directly imported used into a python module.
$(TGTDIR)/%.so: $(OBJDIR)/%.py.so $(OBJDIR)/libmuseqa.a
	$(NVCC) -shared -L$(OBJDIR) -lmuseqa $< -o $@

$(TGTDIR)/%.py: $(SRCDIR)/python/%.py
	@mkdir -p $(dir $@)
	cp -l $< $@

$(OBJDIR)/%.cxx: $(SRCDIR)/python/%.pyx $(INCDIR)/python/*.pxd
	$(PYXC) $(PYXCFLAGS) $< -o $@

$(OBJDIR)/%.py.so: $(OBJDIR)/%.cxx
	$(PYPP) $(PYPPFLAGS) -MMD -c $< -o $@

$(OBJDIR)/libmuseqa.a: $(OBJDIR)/cuda.a
$(OBJDIR)/libmuseqa.a: $(OBJDIR)/encoder.a
$(OBJDIR)/libmuseqa.a: $(OBJDIR)/pairwise/table.a
$(OBJDIR)/libmuseqa.a: $(OBJDIR)/pairwise/database.a
$(OBJDIR)/libmuseqa.a: $(OBJDIR)/pairwise/pairwise.a
$(OBJDIR)/libmuseqa.a: $(OBJDIR)/io/loader/database.a
$(OBJDIR)/libmuseqa.a: $(OBJDIR)/io/loader/parser/fasta.a
$(OBJDIR)/libmuseqa.a: $(OBJDIR)/pairwise/needleman/needleman.a
$(OBJDIR)/libmuseqa.a: $(OBJDIR)/pairwise/needleman/impl/hybrid.a
$(OBJDIR)/libmuseqa.a: $(OBJDIR)/pairwise/needleman/impl/sequential.a
$(OBJDIR)/libmuseqa.a: $(STATICFILES)
	ar rcs $@ $^

.PHONY: all install production debug testing clean

.PRECIOUS: $(OBJDIR)/%.cxx $(OBJDIR)/%.a $(OBJDIR)/%.py.so
