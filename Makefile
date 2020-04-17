SHELL := /bin/bash

NVCC=nvcc
NVCC_ARGS=-lmpi -lineinfo
COMPILER_ARGS=-g -Wno-format -fopenmp

MPI_ARGS_REMOTE=--mca mpi_yield_when_idle 1 -npernode 1 --hostfile hostfile
MPI_ARGS_LOCAL=--mca mpi_yield_when_idle 1 -n 2

SRC=main.cu framework.cu kernels.cu utilities.cpp
TARGET=parallelFramework

all: out

out: $(SRC)
	$(NVCC) $(NVCC_ARGS) -Xcompiler "$(COMPILER_ARGS)" $(SRC) -o $(TARGET)

run: out
	mpiexec $(MPI_ARGS_LOCAL) $(TARGET)

run-remote: out
	mpiexec $(MPI_ARGS_REMOTE) $(TARGET)

clean:
	rm -f $(TARGET)
