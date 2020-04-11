SHELL := /bin/bash

NVCC=nvcc
NVCC_ARGS=-lmpi -lineinfo
COMPILER_ARGS=-fopenmp

SRC=main.cu framework.cu kernels.cu
TARGET=parallelFramework

DEBUG_LEVEL?=1

all: out

out: $(SRC)
	$(NVCC) $(NVCC_ARGS) -Xcompiler $(COMPILER_ARGS) $(SRC) -o $(TARGET)

run: out
	mpiexec -n 1 $(TARGET)

run-remote: $(TARGET)
	mpiexec -n 1 $(TARGET) remote

clean:
	rm -f $(TARGET)
