NAME = msa

SDIR = src
ODIR = obj
MPILKDIR = /usr/lib/openmpi/lib

NVCC = nvcc
MPCC = mpicc
MPPP = mpic++

MPFLAGS = -Wall -std=c++17 -I./$(SDIR)
MCFLAGS = -Wall -std=c99 -lm -I./$(SDIR)
NVFLAGS = -std=c++11 -arch sm_20 -lmpi -lcuda -lcudart -w -I./$(SDIR)
NVLINKFLAGS = -L$(MPILKDIR) -lmpi_cxx -lmpi

NVFILES := $(shell find $(SDIR) -name '*.cu')
MPFILES := $(shell find $(SDIR) -name '*.cpp')
MCFILES := $(shell find $(SDIR) -name '*.c')
DEPS = $(NVFILES:src/%.cu=obj/%.o) $(MPFILES:src/%.cpp=obj/%.o) $(MCFILES:src/%.c=obj/%.o)

.phony: all clean install

all: $(NAME)

$(NAME): $(DEPS)
	$(NVCC) $^ -o $@ $(NVFLAGS) $(NVLINKFLAGS)

$(ODIR)/%.o: src/%.cu $(NVFILES)
	$(NVCC) -c $< -o $@ $(NVFLAGS)

$(ODIR)/%.o: src/%.c $(MCFILES)
	$(MPCC) -c $< -o $@ $(MCFLAGS)

$(ODIR)/%.o: src/%.cpp $(MPFILES)
	$(MPPP) -c $< -o $@ $(MPFLAGS)

install:
	@mkdir -p obj/pairwise

clean:
	@rm -rf $(ODIR)/*.o $(SDIR)/*~ *~ $(NAME) $(ODIR)/pairwise/*.o
