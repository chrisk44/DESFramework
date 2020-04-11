SHELL := /bin/bash

NVCC=nvcc
NVCC_ARGS=-lmpi -lineinfo
COMPILER_ARGS=-fopenmp -g -Wno-format

MPI_ARGS=--mca mpi_yield_when_idle 1 -n 1

SRC=main.cu framework.cu kernels.cu utilities.cpp
TARGET=parallelFramework

DEBUG_LEVEL?=1

all: out

out: $(SRC)
	$(NVCC) $(NVCC_ARGS) -Xcompiler "$(COMPILER_ARGS)" $(SRC) -o $(TARGET)

run: out
	mpiexec $(MPI_ARGS) $(TARGET)

run-remote: $(TARGET)
	mpiexec $(MPI_ARGS) $(TARGET) remote

clean:
	rm -f $(TARGET)