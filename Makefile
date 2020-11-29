SHELL := /bin/bash

NVCC=nvcc
NVCC_ARGS=-lmpi -lineinfo -arch=sm_61 --ptxas-options=-v --maxrregcount 64
COMPILER_ARGS=-g -Wno-format -fopenmp -O3

N?=2

MPI_ARGS=--bind-to none
MPI_ARGS_REMOTE=-npernode 1 --hostfile hostfile #--mca mpi_yield_when_idle 1 # Not needed when using MMPI_Recv
MPI_ARGS_LOCAL=-n $(N) # --mca mpi_yield_when_idle 1

SRC=main.cu framework.cu kernels.cu utilities.cpp
TARGET=parallelFramework

all: out

out: $(SRC)
	$(NVCC) $(NVCC_ARGS) -Xcompiler "$(COMPILER_ARGS)" $(SRC) -o $(TARGET)

run: out
	mpiexec $(MPI_ARGS_LOCAL) $(MPI_ARGS) $(TARGET)

run-remote: out
	mpiexec $(MPI_ARGS_REMOTE) $(MPI_ARGS) $(TARGET)

clean:
	rm -f $(TARGET)
