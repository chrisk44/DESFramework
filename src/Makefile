SHELL := /bin/bash

NVCC=nvcc
NVCC_ARGS=-lmpi -lineinfo -arch=sm_61 --ptxas-options=-v --maxrregcount 64 -lnvidia-ml
COMPILER_ARGS=-g -Wno-format -fopenmp -O3 -DDES_DIRECT_COMPILATION

N?=2

MPI_ARGS=--bind-to none
MPI_ARGS_REMOTE=-npernode 1 --hostfile hostfile #--mca mpi_yield_when_idle 1 # Not needed when using MMPI_Recv
MPI_ARGS_LOCAL=-n $(N) # --mca mpi_yield_when_idle 1

MODELS_SRC=models/mogi/mogi1.cpp models/mogi/mogi2.cpp models/okada/okada1.cpp models/okada/okada2.cpp models/okada/okada3.cpp
DES_SRC=des/computeThread.cu des/coordinatorThread.cpp des/cpuKernel.cpp des/framework.cpp des/gpuKernel.cu des/masterProcess.cpp des/messaging.cpp des/slaveProcess.cpp des/stopwatch.cpp des/utilities.cpp
SRC=main.cu $(MODELS_SRC) $(DES_SRC)
TARGET=parallelFramework

all: out

out: $(SRC)
	$(NVCC) $(NVCC_ARGS) -x cu -Xcompiler "$(COMPILER_ARGS)" $(SRC) -o $(TARGET)

run: out
	mpiexec $(MPI_ARGS_LOCAL) $(MPI_ARGS) $(TARGET)

run-remote: out
	mpiexec $(MPI_ARGS_REMOTE) $(MPI_ARGS) $(TARGET)

clean:
	rm -f $(TARGET)
