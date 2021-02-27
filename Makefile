SHELL := /bin/bash

NVCC=nvcc
NVCC_ARGS=-lmpi -lineinfo -arch=sm_61 --maxrregcount 64 -lnvidia-ml #--ptxas-options=-v
COMPILER_ARGS=-g -Wno-format -fopenmp -O3

N?=2

MPI_ARGS=--bind-to none
MPI_ARGS_REMOTE=-npernode 1 --hostfile hostfile #--mca mpi_yield_when_idle 1 # Not needed when using MMPI_Recv
MPI_ARGS_LOCAL=-n $(N) # --mca mpi_yield_when_idle 1

TARGET=parallelFramework
OBJS = main.o framework.o kernels.o utilities.o masterProcess.o coordinatorThread.o slaveProcess.o computeThread.o

%.o: %.c*
	$(NVCC) $(NVCC_ARGS) -Xcompiler "$(COMPILER_ARGS)" --compile -x cu $^ -o $@

all: $(OBJS)
	nvcc -lmpi -lineinfo -lnvidia-ml -Xcompiler -fopenmp $(OBJS) -o $(TARGET)

run: out
	mpiexec $(MPI_ARGS_LOCAL) $(MPI_ARGS) $(TARGET)

run-remote: out
	mpiexec $(MPI_ARGS_REMOTE) $(MPI_ARGS) $(TARGET)

clean:
	rm -f $(TARGET) *.o
