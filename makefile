NAME = msa

SDIR = src
ODIR = obj
LDIR = /usr/lib/openmpi/lib

NVCC = nvcc
MPCC = mpicc
MPPP = mpic++

MPFLAGS = -Wall -pedantic -std=c++14
NVFLAGS = -arch sm_20 -lmpi -lcuda -lcudart -w
NVLINKFLAGS = -L$(LDIR) -lmpi_cxx -lmpi

NVFILES := $(shell find $(SDIR) -name '*.cu')
MPFILES := $(shell find $(SDIR) -name '*.cpp')
MCFILES := $(shell find $(SDIR) -name '*.c')
DEPS = $(NVFILES:src/%.cu=obj/%.o) $(MPFILES:src/%.cpp=obj/%.o) $(MCFILES:src/%.c=obj/%.o)

all: $(NAME)

$(NAME): $(DEPS)
	$(NVCC) $^ -o $@ $(NVFLAGS) $(NVLINKFLAGS)

$(ODIR)/%.o: src/%.cu $(NVFILES)
	$(NVCC) -c $< -o $@ $(NVFLAGS)

$(ODIR)/%.o: src/%.c $(MCFILES)
	$(MPCC) -c $< -o $@ $(MPFLAGS)

$(ODIR)/%.o: src/%.cpp $(MPFILES)
	$(MPPP) -c $< -o $@ $(MPFLAGS)

.phony: clean

clean:
	rm -rf $(ODIR)/*.o $(SDIR)/*~ *~ $(NAME)