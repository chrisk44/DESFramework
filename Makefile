SHELL := /bin/bash

NVCC=nvcc
NVCC_ARGS=-lmpi -lineinfo -arch=sm_61 --ptxas-options=-v --maxrregcount 62
COMPILER_ARGS=-g -Wno-format -fopenmp -O3

N?=2

MPI_ARGS=--bind-to none
MPI_ARGS_REMOTE=-npernode 1 --hostfile hostfile #--mca mpi_yield_when_idle 1 # Not needed when using MMPI_Recv
MPI_ARGS_LOCAL=-n $(N) # --mca mpi_yield_when_idle 1

APP_ARGS=./data/mogi2_inputs/displ.txt ./data/mogi2_inputs/grid4/grid4.txt 2

SRC=main.cu framework.cu kernels.cu utilities.cpp
TARGET=parallelFramework

all: out

out: $(SRC)
	$(NVCC) $(NVCC_ARGS) -Xcompiler "$(COMPILER_ARGS)" $(SRC) -o $(TARGET)

run: out
	mpiexec $(MPI_ARGS_LOCAL) $(MPI_ARGS) $(TARGET) $(APP_ARGS)

run-remote: out
	mpiexec $(MPI_ARGS_REMOTE) $(MPI_ARGS) $(TARGET) $(APP_ARGS)

clean:
	rm -f $(TARGET)
