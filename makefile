NAME = msa

INCDIR = inc
SRCDIR = src
OBJDIR = obj

MPCC = mpicc
MPPP = mpic++
NVCC = nvcc

# Target architecture for CUDA compilation. This indicates the minimum
# support required for the codebase.
NVARCH ?= sm_30

# Defining language standards to be used. These can be overriden by
# environment variables.
STDC   ?= c99
STDCPP ?= c++14
STDCU  ?= c++11

MPILIBDIR = /usr/lib/openmpi/lib

NVCCSPECIAL = --relocatable-device-code=true -D_MWAITXINTRIN_H_INCLUDED

MPCCFLAGS = -std=$(STDC) -I$(INCDIR) -g -Wall -lm
MPPPFLAGS = -std=$(STDCPP) -I$(INCDIR) -g -Wall
NVCCFLAGS = -std=$(STDCU) -I$(INCDIR) -g -arch $(NVARCH) -lmpi -lcuda -lcudart -w $(NVCCSPECIAL)
LINKFLAGS = -L$(MPILIBDIR) -lmpi_cxx -lmpi -g

# Lists all files to be compiled and separates them according to their
# corresponding compilers.
MPCCFILES := $(shell find $(SRCDIR) -name '*.c')
MPPPFILES := $(shell find $(SRCDIR) -name '*.cpp')
NVCCFILES := $(shell find $(SRCDIR) -name '*.cu')

SRCINTERNAL = $(sort $(dir $(wildcard $(SRCDIR)/*/. $(SRCDIR)/*/*/.)))
OBJINTERNAL = $(SRCINTERNAL:$(SRCDIR)/%=$(OBJDIR)/%)

ODEPS = $(MPCCFILES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)                        \
        $(MPPPFILES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)                      \
        $(NVCCFILES:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)
HDEPS = $(ODEPS:$(OBJDIR)/%.o=$(OBJDIR)/%.d)

.phony: all clean install

all: install $(NAME)

$(NAME): $(ODEPS)
	$(NVCC) $(LINKFLAGS) $^ -o $@

# Creates dependency on header files. This is valuable so that whenever
# a header file is changed, all objects depending on it will be recompiled.
-include $(HDEPS)

# Compiling C files.
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(MPCC) $(MPCCFLAGS) -MMD -c $< -o $@

# Compiling C++ files.
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(MPPP) $(MPPPFLAGS) -MMD -c $< -o $@

# Compiling CUDA files.
$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	@$(NVCC) $(NVCCFLAGS) -M $< -odir $(OBJDIR) > $(@:%.o=%.d)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

install:
	@mkdir -p $(OBJINTERNAL)

clean:
	@rm -rf $(NAME) $(OBJDIR) $(SRCDIR)/*~ *~
