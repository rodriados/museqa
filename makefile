NAME = msa

SDIR = src
ODIR = obj

NVCC = nvcc
MPCC = mpicc
MPPP = mpic++

MPILKDIR = /usr/lib/openmpi/lib

MPFLAGS = -Wall -std=c++17 -I./$(SDIR)
MCFLAGS = -Wall -std=c99 -lm -I./$(SDIR)
NVFLAGS = -std=c++11 -arch sm_20 -lmpi -lcuda -lcudart -w -I./$(SDIR)
NVLINKFLAGS = -L$(MPILKDIR) -lmpi_cxx -lmpi

NVFILES := $(shell find $(SDIR) -name '*.cu')
MPFILES := $(shell find $(SDIR) -name '*.cpp')
MCFILES := $(shell find $(SDIR) -name '*.c')

DEPS = $(NVFILES:src/%.cu=obj/%.o) $(MPFILES:src/%.cpp=obj/%.o) $(MCFILES:src/%.c=obj/%.o)
HDEPS = $(DEPS:obj/%.o=obj/%.d)

.phony: all clean install

all: $(NAME)

$(NAME): $(DEPS)
	$(NVCC) $(NVFLAGS) $(NVLINKFLAGS) $^ -o $@

-include $(HDEPS)

$(ODIR)/%.o: src/%.cu
	@$(NVCC) $(NVFLAGS) -M $< -odir obj > $(@:%.o=%.d)
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(ODIR)/%.o: src/%.c
	$(MPCC) $(MCFLAGS) -MMD -c $< -o $@

$(ODIR)/%.o: src/%.cpp
	$(MPPP) $(MPFLAGS) -MMD -c $< -o $@

install:
	@mkdir -p obj/pairwise

clean:
	@rm -rf $(DEPS) $(HDEPS) $(NAME) $(SDIR)/*~ *~
